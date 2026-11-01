//===- RISCVRedundantInstrElim.cpp - Remove redundant instructions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Eliminate redundant copies and constant materializations after register
// allocation. Track equivalence classes of physical registers, together with
// known integer constants. The meet operation retains only equalities that
// hold on every incoming edge.
//
// Value numbers are local to a dataflow state, not instruction identities. In
// particular, executing the same definition on two iterations of a loop does
// not imply that it computes the same value. First solve the dataflow problem
// to a fixed point, then remove instructions using the converged states.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <numeric>
#include <optional>
#include <queue>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "riscv-redundant-instr-elim"
#define RISCV_REDUNDANT_INSTR_ELIM_NAME                                        \
  "RISC-V Redundant Instruction Elimination"

STATISTIC(NumCopiesRemoved, "Number of redundant RISC-V copies removed");
STATISTIC(NumConstantsRemoved,
          "Number of redundant RISC-V constant instructions removed");

namespace {

struct ValueInfo {
  unsigned Number;
  std::optional<int64_t> Constant;
};

/// A partition of the tracked locations. Constants form a flat lattice: a
/// location either has one known XLEN-bit value, or has no known constant.
class AvailableValues {
  SmallVector<unsigned, 96> Numbers;
  SmallVector<std::optional<int64_t>, 96> Constants;
  // Every value number in use is below NextNumber.
  unsigned NextNumber;

public:
  explicit AvailableValues(unsigned Size)
      : Numbers(Size), Constants(Size), NextNumber(Size) {
    std::iota(Numbers.begin(), Numbers.end(), 0);
    // Location zero is always x0, including in an otherwise unknown state.
    Constants[0] = 0;
  }

  ValueInfo get(unsigned I) const { return {Numbers[I], Constants[I]}; }
  bool equal(unsigned A, unsigned B) const { return Numbers[A] == Numbers[B]; }

  void forget(unsigned I) {
    Numbers[I] = NextNumber++;
    Constants[I].reset();
  }

  void assign(unsigned I, ValueInfo Value) {
    Numbers[I] = Value.Number;
    Constants[I] = Value.Constant;
  }

  void assignConstant(unsigned I, int64_t Constant) {
    for (unsigned J = 0, E = Numbers.size(); J != E; ++J) {
      if (Constants[J] == Constant) {
        assign(I, get(J));
        return;
      }
    }
    Numbers[I] = NextNumber++;
    Constants[I] = Constant;
  }

  /// Refine an edge on which A == B. Return false for an infeasible edge.
  bool assumeEqual(unsigned A, unsigned B) {
    if (Constants[A] && Constants[B] && Constants[A] != Constants[B])
      return false;
    if (equal(A, B))
      return true;
    unsigned AN = Numbers[A], BN = Numbers[B];
    auto Constant = Constants[A] ? Constants[A] : Constants[B];
    for (unsigned I = 0, E = Numbers.size(); I != E; ++I) {
      if (Numbers[I] == AN || Numbers[I] == BN) {
        Numbers[I] = AN;
        Constants[I] = Constant;
      }
    }
    return true;
  }

  void canonicalize() {
    // Number each class after its first location.
    constexpr unsigned None = ~0u;
    SmallVector<unsigned, 128> Classes(NextNumber, None);
    for (unsigned I = 0, E = Numbers.size(); I != E; ++I) {
      unsigned &Class = Classes[Numbers[I]];
      if (Class == None)
        Class = I;
      Numbers[I] = Class;
    }
    NextNumber = Numbers.size();
  }

  void meet(const AvailableValues &Other) {
    DenseMap<std::pair<unsigned, unsigned>, unsigned> Classes;
    for (unsigned I = 0, E = Numbers.size(); I != E; ++I) {
      Numbers[I] =
          Classes.try_emplace(std::make_pair(Numbers[I], Other.Numbers[I]), I)
              .first->second;
      if (Constants[I] != Other.Constants[I])
        Constants[I].reset();
    }
    NextNumber = Numbers.size();
  }

  bool operator==(const AvailableValues &Other) const {
    return Numbers == Other.Numbers && Constants == Other.Constants;
  }
};

struct RegisterRange {
  unsigned First;
  unsigned Size;
};

class RISCVRedundantInstrElimImpl {
  MachineFunction &MF;
  const RISCVSubtarget &ST;
  const RISCVInstrInfo &TII;
  const RISCVRegisterInfo &TRI;

  // x0-x31.
  SmallVector<MCPhysReg, 96> Registers;
  DenseMap<Register, unsigned> RegisterNumbers;
  // The tracked register containing each register unit. Zero means x0, which
  // is never clobbered, or a unit outside the tracked registers.
  SmallVector<unsigned> RegUnitLocations;
  DenseMap<const MachineBasicBlock *, AvailableValues> LiveOutValues;

  std::optional<RegisterRange> getRegisterRange(const MachineOperand &MO) const;
  std::optional<int64_t> getConstant(const MachineOperand &MO,
                                     const AvailableValues &Values) const;
  std::optional<int64_t> evaluateConstant(const MachineInstr &MI,
                                          const AvailableValues &Values) const;
  std::optional<RegisterRange> getCopySource(const MachineInstr &MI,
                                             RegisterRange Dst) const;
  std::optional<AvailableValues>
  getIncomingValues(MachineBasicBlock &MBB) const;
  bool refineEdge(MachineBasicBlock &Pred, MachineBasicBlock &Succ,
                  AvailableValues &Values) const;
  void processInstruction(MachineInstr &MI, AvailableValues &Values,
                          SmallVectorImpl<MachineInstr *> *Redundant = nullptr);
  void solve();

public:
  explicit RISCVRedundantInstrElimImpl(MachineFunction &MF);
  bool run();
};

class RISCVRedundantInstrElimLegacy : public MachineFunctionPass {
public:
  static char ID;
  RISCVRedundantInstrElimLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return RISCV_REDUNDANT_INSTR_ELIM_NAME;
  }
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs().setNoPHIs();
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    return RISCVRedundantInstrElimImpl(MF).run();
  }
};

} // end anonymous namespace

static bool isCSRInstruction(unsigned Opcode) {
  switch (Opcode) {
  default:
    return false;
  case RISCV::CSRRW:
  case RISCV::CSRRWI:
  case RISCV::CSRRS:
  case RISCV::CSRRSI:
  case RISCV::CSRRC:
  case RISCV::CSRRCI:
    return true;
  }
}

static bool canModelValue(const MachineInstr &MI) {
  return !MI.isBundled() && !MI.isCall() && !MI.isTerminator() &&
         !MI.isInlineAsm() && !MI.isMetaInstruction() && !MI.isConvergent() &&
         !MI.isNotDuplicable() && !MI.hasUnmodeledSideEffects() &&
         !MI.hasOrderedMemoryRef() && !MI.getHeapAllocMarker() &&
         !MI.getPCSections() && !MI.getMMRAMetadata() &&
         !llvm::any_of(MI.operands(), [](const MachineOperand &MO) {
           return MO.isRegMask() ||
                  (MO.isReg() && MO.isDef() && MO.isImplicit());
         });
}

RISCVRedundantInstrElimImpl::RISCVRedundantInstrElimImpl(MachineFunction &MF)
    : MF(MF), ST(MF.getSubtarget<RISCVSubtarget>()), TII(*ST.getInstrInfo()),
      TRI(*ST.getRegisterInfo()) {
  Registers.resize(32);
  for (MCPhysReg Reg : RISCV::GPRRegClass)
    Registers[TRI.getEncodingValue(Reg)] = Reg;
  RegUnitLocations.resize(TRI.getNumRegUnits());
  for (auto [I, Reg] : llvm::enumerate(Registers)) {
    RegisterNumbers.try_emplace(Reg, I);
    for (MCRegUnit Unit : TRI.regunits(Reg)) {
      assert(!RegUnitLocations[static_cast<unsigned>(Unit)] &&
             "Tracked registers must not overlap");
      RegUnitLocations[static_cast<unsigned>(Unit)] = I;
    }
  }
}

std::optional<RegisterRange>
RISCVRedundantInstrElimImpl::getRegisterRange(const MachineOperand &MO) const {
  if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
      MO.isUndef())
    return std::nullopt;
  Register Reg = MO.getReg();
  if (auto I = RegisterNumbers.find(Reg); I != RegisterNumbers.end())
    return RegisterRange{I->second, 1};
  if (RISCV::GPRPairRegClass.contains(Reg) && Reg != RISCV::X0_Pair)
    return RegisterRange{TRI.getEncodingValue(Reg), 2};
  return std::nullopt;
}

std::optional<int64_t>
RISCVRedundantInstrElimImpl::getConstant(const MachineOperand &MO,
                                         const AvailableValues &Values) const {
  if (MO.isImm())
    return MO.getImm();
  auto Reg = getRegisterRange(MO);
  if (!Reg || Reg->First >= 32 || Reg->Size != 1)
    return std::nullopt;
  return Values.get(Reg->First).Constant;
}

std::optional<int64_t> RISCVRedundantInstrElimImpl::evaluateConstant(
    const MachineInstr &MI, const AvailableValues &Values) const {
  unsigned Opcode = MI.getOpcode();
  if (Opcode == RISCV::LUI && MI.getOperand(1).isImm())
    return SignExtend64<32>(uint64_t(MI.getOperand(1).getImm()) << 12);
  if ((Opcode == RISCV::PseudoLI || Opcode == RISCV::QC_LI ||
       Opcode == RISCV::QC_E_LI) &&
      MI.getOperand(1).isImm())
    return APInt(ST.getXLen(), MI.getOperand(1).getImm(), /*isSigned=*/true,
                 /*implicitTrunc=*/true)
        .getSExtValue();

  // Evaluate only pure integer operations with defined, known operands. APInt
  // gives RV32/RV64 wrapping and word sign-extension without host overflow.
  bool Word = false;
  switch (Opcode) {
  default:
    return std::nullopt;
  case RISCV::ADDIW:
  case RISCV::ADDW:
  case RISCV::SUBW:
  case RISCV::SLLIW:
  case RISCV::SRLIW:
  case RISCV::SRAIW:
    Word = true;
    [[fallthrough]];
  case RISCV::ADDI:
  case RISCV::ADD:
  case RISCV::SUB:
  case RISCV::ANDI:
  case RISCV::AND:
  case RISCV::ORI:
  case RISCV::OR:
  case RISCV::XORI:
  case RISCV::XOR:
  case RISCV::SLLI:
  case RISCV::SRLI:
  case RISCV::SRAI:
    break;
  }
  auto LHS = getConstant(MI.getOperand(1), Values);
  auto RHS = getConstant(MI.getOperand(2), Values);
  if (!LHS || !RHS)
    return std::nullopt;
  unsigned Bits = Word ? 32 : ST.getXLen();
  APInt A(Bits, *LHS, /*isSigned=*/true, /*implicitTrunc=*/true);
  APInt B(Bits, *RHS, /*isSigned=*/true, /*implicitTrunc=*/true);
  APInt Result(Bits, 0);
  switch (Opcode) {
  default:
    llvm_unreachable("Unexpected constant instruction");
  case RISCV::ADDI:
  case RISCV::ADD:
  case RISCV::ADDIW:
  case RISCV::ADDW:
    Result = A + B;
    break;
  case RISCV::SUB:
  case RISCV::SUBW:
    Result = A - B;
    break;
  case RISCV::ANDI:
  case RISCV::AND:
    Result = A & B;
    break;
  case RISCV::ORI:
  case RISCV::OR:
    Result = A | B;
    break;
  case RISCV::XORI:
  case RISCV::XOR:
    Result = A ^ B;
    break;
  case RISCV::SLLI:
  case RISCV::SLLIW:
    Result = A.shl(*RHS & (Bits - 1));
    break;
  case RISCV::SRLI:
  case RISCV::SRLIW:
    Result = A.lshr(*RHS & (Bits - 1));
    break;
  case RISCV::SRAI:
  case RISCV::SRAIW:
    Result = A.ashr(*RHS & (Bits - 1));
    break;
  }
  return Result.getSExtValue();
}

std::optional<RegisterRange>
RISCVRedundantInstrElimImpl::getCopySource(const MachineInstr &MI,
                                           RegisterRange Dst) const {
  auto Copy = TII.isCopyInstr(MI);
  if (!Copy)
    return std::nullopt;
  auto Src = getRegisterRange(*Copy->Source);
  if (!Src || Src->Size != Dst.Size)
    return std::nullopt;
  // In-register FP formats may truncate or sign-extend an XLEN-bit value.
  switch (MI.getOpcode()) {
  case RISCV::FSGNJ_H_INX:
    return std::nullopt;
  case RISCV::FSGNJ_S_INX:
    if (ST.is64Bit())
      return std::nullopt;
    break;
  default:
    break;
  }
  return Src;
}

bool RISCVRedundantInstrElimImpl::refineEdge(MachineBasicBlock &Pred,
                                             MachineBasicBlock &Succ,
                                             AvailableValues &Values) const {
  if (Pred.succ_size() != 2)
    return true;
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (TII.analyzeBranch(Pred, TBB, FBB, Cond, /*AllowModify=*/false) ||
      Cond.size() != 3 || !TBB ||
      (Cond[0].getImm() != RISCV::BEQ && Cond[0].getImm() != RISCV::BNE) ||
      ((Cond[0].getImm() == RISCV::BEQ) != (TBB == &Succ)))
    return true;
  auto LHS = getRegisterRange(Cond[1]);
  auto RHS = getRegisterRange(Cond[2]);
  if (LHS && RHS && LHS->First < 32 && RHS->First < 32 && LHS->Size == 1 &&
      RHS->Size == 1)
    return Values.assumeEqual(LHS->First, RHS->First);
  return true;
}

std::optional<AvailableValues>
RISCVRedundantInstrElimImpl::getIncomingValues(MachineBasicBlock &MBB) const {
  std::optional<AvailableValues> Incoming;
  if (&MBB == &MF.front() || MBB.isEHPad() || MBB.hasAddressTaken())
    Incoming.emplace(Registers.size());
  for (MachineBasicBlock *Pred : MBB.predecessors()) {
    auto I = LiveOutValues.find(Pred);
    if (I == LiveOutValues.end())
      continue;
    AvailableValues Edge = I->second;
    if (!refineEdge(*Pred, MBB, Edge))
      continue;
    if (Incoming)
      Incoming->meet(Edge);
    else
      Incoming = std::move(Edge);
  }
  // No reachable predecessor is the bottom element, distinct from a state
  // with no known equalities. Only reachable states participate in the meet.
  return Incoming;
}

void RISCVRedundantInstrElimImpl::processInstruction(
    MachineInstr &MI, AvailableValues &Values,
    SmallVectorImpl<MachineInstr *> *Redundant) {
  if (MI.isDebugInstr())
    return;
  std::optional<RegisterRange> Dst;
  std::optional<int64_t> Constant;
  SmallVector<ValueInfo, 8> Result;
  bool ModelValue = canModelValue(MI);
  if (ModelValue && MI.getNumExplicitDefs() == 1) {
    Dst = getRegisterRange(MI.getOperand(0));
    if (Dst && Dst->First != 0) {
      if (Dst->First < 32 && Dst->Size == 1)
        Constant = evaluateConstant(MI, Values);
      if (!Constant) {
        if (auto Src = getCopySource(MI, *Dst))
          for (unsigned I = 0; I != Src->Size; ++I)
            Result.push_back(Values.get(Src->First + I));
      }
    }
  }

  bool Same = Constant && Dst && Values.get(Dst->First).Constant == Constant;
  if (!Result.empty()) {
    Same = true;
    for (auto [I, Value] : llvm::enumerate(Result))
      Same &= Values.get(Dst->First + I).Number == Value.Number;
  }
  if (Redundant && Same && !MI.getFlag(MachineInstr::FrameSetup) &&
      !MI.getFlag(MachineInstr::FrameDestroy)) {
    Redundant->push_back(&MI);
    if (Constant)
      ++NumConstantsRemoved;
    else
      ++NumCopiesRemoved;
    return;
  }

  bool HasRegMask = llvm::any_of(
      MI.operands(), [](const MachineOperand &MO) { return MO.isRegMask(); });
  // Calls and inline assembly describe their register effects with operands.
  // Other unmodeled effects and calls without a mask are opaque barriers.
  bool Opaque = MI.hasUnmodeledSideEffects() && !MI.isCall() &&
                !MI.isInlineAsm() && !isCSRInstruction(MI.getOpcode());
  bool ClobberAll = Opaque || MI.isBundled() || (MI.isCall() && !HasRegMask);
  if (ClobberAll) {
    for (unsigned I = 1, E = Registers.size(); I != E; ++I)
      Values.forget(I);
  } else {
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isRegMask()) {
        for (unsigned I = 1, E = Registers.size(); I != E; ++I)
          if (MO.clobbersPhysReg(Registers[I]))
            Values.forget(I);
      } else if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical()) {
        for (MCRegUnit Unit : TRI.regunits(MO.getReg()))
          if (unsigned I = RegUnitLocations[static_cast<unsigned>(Unit)])
            Values.forget(I);
      }
    }
  }

  if (Constant)
    Values.assignConstant(Dst->First, *Constant);
  for (auto [I, Value] : llvm::enumerate(Result))
    Values.assign(Dst->First + I, Value);
}

void RISCVRedundantInstrElimImpl::solve() {
  std::queue<MachineBasicBlock *> Worklist;
  SmallPtrSet<MachineBasicBlock *, 32> Queued;
  auto Enqueue = [&](MachineBasicBlock *MBB) {
    if (Queued.insert(MBB).second)
      Worklist.push(MBB);
  };
  for (MachineBasicBlock *MBB :
       ReversePostOrderTraversal<MachineFunction *>(&MF))
    Enqueue(MBB);
  for (MachineBasicBlock &MBB : MF)
    if (MBB.isEHPad() || MBB.hasAddressTaken())
      Enqueue(&MBB);
  while (!Worklist.empty()) {
    MachineBasicBlock &MBB = *Worklist.front();
    Worklist.pop();
    Queued.erase(&MBB);
    auto Incoming = getIncomingValues(MBB);
    if (!Incoming)
      continue;
    AvailableValues Values = std::move(*Incoming);
    for (MachineInstr &MI : MBB.instrs())
      processInstruction(MI, Values);
    Values.canonicalize();
    auto I = LiveOutValues.find(&MBB);
    if (I != LiveOutValues.end() && I->second == Values)
      continue;
    LiveOutValues.insert_or_assign(&MBB, std::move(Values));
    for (MachineBasicBlock *Succ : MBB.successors())
      Enqueue(Succ);
  }
}

bool RISCVRedundantInstrElimImpl::run() {
  // Outlined helpers can have an implicit register interface that is not
  // represented by their operands. Match MachineCopyPropagation's restriction.
  if (MF.empty() || MF.isOutlined())
    return false;
  solve();
  SmallVector<MachineInstr *, 16> Redundant;
  for (MachineBasicBlock &MBB : MF) {
    if (!LiveOutValues.contains(&MBB))
      continue;
    auto Values = getIncomingValues(MBB);
    assert(Values && "Expected a reachable block");
    for (MachineInstr &MI : MBB.instrs())
      processInstruction(MI, *Values, &Redundant);
  }
  if (Redundant.empty())
    return false;
  for (MachineInstr *MI : Redundant) {
    LLVM_DEBUG(dbgs() << "Removing redundant instruction: " << *MI);
    MI->eraseFromParent();
  }
  if (MF.getRegInfo().tracksLiveness()) {
    SmallVector<MachineBasicBlock *, 16> Blocks;
    for (MachineBasicBlock &MBB : llvm::reverse(MF))
      Blocks.push_back(&MBB);
    fullyRecomputeLiveIns(Blocks);
    for (MachineBasicBlock &MBB : MF)
      recomputeLivenessFlags(MBB);
  }
  return true;
}

char RISCVRedundantInstrElimLegacy::ID = 0;
INITIALIZE_PASS(RISCVRedundantInstrElimLegacy, DEBUG_TYPE,
                RISCV_REDUNDANT_INSTR_ELIM_NAME, false, false)

FunctionPass *llvm::createRISCVRedundantInstrElimLegacyPass() {
  return new RISCVRedundantInstrElimLegacy();
}

PreservedAnalyses
RISCVRedundantInstrElimPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  if (!RISCVRedundantInstrElimImpl(MF).run())
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
