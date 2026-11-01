//===- RISCVRedundantInstrElim.cpp - Remove redundant instructions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Eliminate redundant copies, constant materializations and spill reloads after
// register allocation. Track equivalence classes of physical registers and
// compiler-owned spill slots, together with known integer constants. The meet
// operation retains only equalities that hold on every incoming edge.
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
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/IR/Function.h"
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
STATISTIC(NumReloadsRemoved,
          "Number of redundant RISC-V spill reloads removed");

namespace {

struct ValueInfo {
  unsigned Number;
  std::optional<int64_t> Constant;

  bool operator==(const ValueInfo &Other) const {
    return Number == Other.Number && Constant == Other.Constant;
  }
  bool operator!=(const ValueInfo &Other) const { return !(*this == Other); }
};

/// A partition of the tracked locations. Constants form a flat lattice: a
/// location either has one known XLEN-bit value, or has no known constant.
class AvailableValues {
  SmallVector<unsigned, 96> Numbers;
  SmallVector<std::optional<int64_t>, 96> Constants;
  // Absent slots have distinct unknown values. Unlike registers, spill slots
  // are tracked lazily, and uninformative entries are discarded at block exits.
  DenseMap<int, ValueInfo> SpillValues;
  // Every value number in use is below NextNumber.
  unsigned NextNumber;

public:
  bool VStartIsZero = false;

  explicit AvailableValues(unsigned Size)
      : Numbers(Size), Constants(Size), NextNumber(Size) {
    std::iota(Numbers.begin(), Numbers.end(), 0);
    // Location zero is always x0, including in an otherwise unknown state.
    Constants[0] = 0;
  }

  ValueInfo get(unsigned I) const { return {Numbers[I], Constants[I]}; }
  bool equal(unsigned A, unsigned B) const { return Numbers[A] == Numbers[B]; }

  ValueInfo getSpill(int FI) {
    auto [I, Inserted] =
        SpillValues.try_emplace(FI, ValueInfo{NextNumber, std::nullopt});
    if (Inserted)
      ++NextNumber;
    return I->second;
  }

  void assignSpill(int FI, ValueInfo Value) { SpillValues[FI] = Value; }
  void forgetSpills() { SpillValues.clear(); }

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
    for (const auto &[FI, Value] : SpillValues) {
      if (Value.Constant == Constant) {
        assign(I, Value);
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
    for (auto &[FI, Value] : SpillValues) {
      if (Value.Number == AN || Value.Number == BN)
        Value = {AN, Constant};
    }
    return true;
  }

  void canonicalize() {
    if (!SpillValues.empty()) {
      // A slot with no constant and no equality to another location is
      // unknown. Forgetting it loses no information, and avoids keeping every
      // slot ever loaded in every subsequent block's state.
      SmallVector<unsigned, 128> Uses(NextNumber);
      for (unsigned Number : Numbers)
        ++Uses[Number];
      for (const auto &[FI, Value] : SpillValues)
        ++Uses[Value.Number];
      SpillValues.remove_if([&](const auto &Entry) {
        return !Entry.second.Constant && Uses[Entry.second.Number] == 1;
      });
    }

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

    // Frame-index order makes value numbers independent of hash-table order.
    SmallVector<int, 8> Slots;
    for (const auto &[FI, Value] : SpillValues)
      Slots.push_back(FI);
    llvm::sort(Slots);
    for (int FI : Slots) {
      ValueInfo &Value = SpillValues.find(FI)->second;
      unsigned &Class = Classes[Value.Number];
      if (Class == None)
        Class = NextNumber++;
      Value.Number = Class;
    }
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
    SpillValues.remove_if([&](const auto &Entry) {
      return !Other.SpillValues.contains(Entry.first);
    });
    for (auto &[FI, Value] : SpillValues) {
      const ValueInfo &OtherValue = Other.SpillValues.find(FI)->second;
      auto [Class, Inserted] = Classes.try_emplace(
          std::make_pair(Value.Number, OtherValue.Number), NextNumber);
      if (Inserted)
        ++NextNumber;
      Value.Number = Class->second;
      if (Value.Constant != OtherValue.Constant)
        Value.Constant.reset();
    }
    VStartIsZero &= Other.VStartIsZero;
  }

  bool operator==(const AvailableValues &Other) const {
    return Numbers == Other.Numbers && Constants == Other.Constants &&
           SpillValues == Other.SpillValues &&
           VStartIsZero == Other.VStartIsZero;
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

  // x0-x31, optionally v0-v31, and optionally the full FLEN-bit FP registers.
  // Narrow FP views cannot be treated as bitwise copies: FSGNJ may check NaN
  // boxing, and writing a narrow FP register also changes its upper bits.
  SmallVector<MCPhysReg, 96> Registers;
  DenseMap<Register, unsigned> RegisterNumbers;
  // The tracked register containing each register unit. Zero means x0, which
  // is never clobbered, or a unit outside the tracked registers.
  SmallVector<unsigned> RegUnitLocations;
  // sp, the frame pointer and the base pointer. Writing one of them changes
  // the address of every spill slot.
  SmallVector<unsigned, 3> StackBaseLocations;
  DenseMap<const MachineBasicBlock *, AvailableValues> LiveOutValues;
  unsigned VectorRegBase = 0;
  unsigned FPRegBase = 0;
  unsigned FPLoadOpcode = 0;
  unsigned FPStoreOpcode = 0;
  unsigned FPRegBytes = 0;

  std::optional<RegisterRange> getRegisterRange(const MachineOperand &MO) const;
  std::optional<int> getSpillSlot(const MachineInstr &MI) const;
  std::optional<int64_t> getConstant(const MachineOperand &MO,
                                     const AvailableValues &Values) const;
  std::optional<int64_t> evaluateConstant(const MachineInstr &MI,
                                          const AvailableValues &Values) const;
  std::optional<RegisterRange>
  getCopySource(const MachineInstr &MI, RegisterRange Dst,
                const AvailableValues &Values) const;
  std::optional<AvailableValues>
  getIncomingValues(MachineBasicBlock &MBB) const;
  bool refineEdge(MachineBasicBlock &Pred, MachineBasicBlock &Succ,
                  AvailableValues &Values) const;
  bool getVStartIsZero(const MachineInstr &MI,
                       const AvailableValues &Values) const;
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

static unsigned getVectorCopySize(unsigned Opcode) {
  switch (Opcode) {
  default:
    return 0;
  case RISCV::VMV1R_V:
    return 1;
  case RISCV::VMV2R_V:
    return 2;
  case RISCV::VMV4R_V:
    return 4;
  case RISCV::VMV8R_V:
    return 8;
  }
}

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
  if (ST.hasVInstructions()) {
    VectorRegBase = Registers.size();
    Registers.resize(VectorRegBase + 32);
    for (MCPhysReg Reg : RISCV::VRRegClass)
      Registers[VectorRegBase + TRI.getEncodingValue(Reg)] = Reg;
  }
  const TargetRegisterClass *FPRC = nullptr;
  if (ST.hasStdExtQ()) {
    FPRC = &RISCV::FPR128RegClass;
    FPLoadOpcode = RISCV::FLQ;
    FPStoreOpcode = RISCV::FSQ;
    FPRegBytes = 16;
  } else if (ST.hasStdExtD()) {
    FPRC = &RISCV::FPR64RegClass;
    FPLoadOpcode = RISCV::FLD;
    FPStoreOpcode = RISCV::FSD;
    FPRegBytes = 8;
  } else if (ST.hasStdExtF()) {
    FPRC = &RISCV::FPR32RegClass;
    FPLoadOpcode = RISCV::FLW;
    FPStoreOpcode = RISCV::FSW;
    FPRegBytes = 4;
  }
  if (FPRC) {
    FPRegBase = Registers.size();
    Registers.resize(FPRegBase + 32);
    for (MCPhysReg Reg : *FPRC)
      Registers[FPRegBase + TRI.getEncodingValue(Reg)] = Reg;
  }
  RegUnitLocations.resize(TRI.getNumRegUnits());
  for (auto [I, Reg] : llvm::enumerate(Registers)) {
    RegisterNumbers.try_emplace(Reg, I);
    for (MCRegUnit Unit : TRI.regunits(Reg)) {
      assert(!RegUnitLocations[static_cast<unsigned>(Unit)] &&
             "Tracked registers must not overlap");
      RegUnitLocations[static_cast<unsigned>(Unit)] = I;
    }
  }
  StackBaseLocations.push_back(RegisterNumbers.lookup(RISCV::X2));
  StackBaseLocations.push_back(
      RegisterNumbers.lookup(TRI.getFrameRegister(MF)));
  if (ST.getFrameLowering()->hasBP(MF))
    StackBaseLocations.push_back(RegisterNumbers.lookup(RISCV::X9));
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
  if (VectorRegBase) {
    unsigned Size = RISCV::VRM2RegClass.contains(Reg)   ? 2
                    : RISCV::VRM4RegClass.contains(Reg) ? 4
                    : RISCV::VRM8RegClass.contains(Reg) ? 8
                                                        : 0;
    if (Size)
      return RegisterRange{VectorRegBase + TRI.getEncodingValue(Reg), Size};
  }
  return std::nullopt;
}

std::optional<int>
RISCVRedundantInstrElimImpl::getSpillSlot(const MachineInstr &MI) const {
  unsigned Opcode = MI.getOpcode();
  unsigned Bytes;
  if (Opcode == (ST.is64Bit() ? RISCV::LD : RISCV::LW) ||
      Opcode == (ST.is64Bit() ? RISCV::SD : RISCV::SW))
    Bytes = ST.getXLen() / 8;
  else if (FPRegBytes && (Opcode == FPLoadOpcode || Opcode == FPStoreOpcode))
    Bytes = FPRegBytes;
  else
    return std::nullopt;
  if (!MI.hasOneMemOperand() || MI.hasUnmodeledSideEffects())
    return std::nullopt;
  const MachineMemOperand &MMO = **MI.memoperands_begin();
  const auto *PSV =
      dyn_cast_or_null<FixedStackPseudoSourceValue>(MMO.getPseudoValue());
  if (!PSV || !MMO.isUnordered() || MMO.isVolatile() || MMO.isAtomic() ||
      MMO.getOffset() || MMO.getSize() != LocationSize::precise(Bytes))
    return std::nullopt;
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  int FI = PSV->getFrameIndex();
  if (FI < MFI.getObjectIndexBegin() || FI >= MFI.getObjectIndexEnd() ||
      MFI.isDeadObjectIndex(FI) || !MFI.isSpillSlotObjectIndex(FI) ||
      MFI.getStackID(FI) != TargetStackID::Default ||
      MFI.getObjectSize(FI) != Bytes || PSV->isAliased(&MFI) ||
      PSV->mayAlias(&MFI))
    return std::nullopt;
  return FI;
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

std::optional<RegisterRange> RISCVRedundantInstrElimImpl::getCopySource(
    const MachineInstr &MI, RegisterRange Dst,
    const AvailableValues &Values) const {
  if (unsigned Size = getVectorCopySize(MI.getOpcode())) {
    auto Src = getRegisterRange(MI.getOperand(1));
    if (Values.VStartIsZero && Src && Dst.Size == Size && Src->Size == Size &&
        VectorRegBase && Dst.First >= VectorRegBase &&
        Dst.First < VectorRegBase + 32 && Src->First >= VectorRegBase &&
        Src->First < VectorRegBase + 32)
      return Src;
    return std::nullopt;
  }
  auto Copy = TII.isCopyInstr(MI);
  // These are bitwise copies only when the integer and FP widths agree.
  if (!Copy && FPRegBytes == ST.getXLen() / 8 &&
      (MI.getOpcode() == RISCV::FMV_X_D || MI.getOpcode() == RISCV::FMV_D_X ||
       MI.getOpcode() == RISCV::FMV_X_W || MI.getOpcode() == RISCV::FMV_W_X))
    Copy = DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
  // isCopyInstr does not currently recognize the Q-extension move.
  if (!Copy && MI.getOpcode() == RISCV::FSGNJ_Q &&
      MI.getOperand(1).isIdenticalTo(MI.getOperand(2)))
    Copy = DestSourcePair{MI.getOperand(0), MI.getOperand(1)};
  if (!Copy)
    return std::nullopt;
  auto Src = getRegisterRange(*Copy->Source);
  if (!Src || Src->Size != Dst.Size)
    return std::nullopt;
  // A cross-bank copy must transfer the complete XLEN/FLEN-bit value.
  bool DstFP = FPRegBase && Dst.First >= FPRegBase;
  bool SrcFP = FPRegBase && Src->First >= FPRegBase;
  bool IntegerFPCopy = (DstFP && Src->First < 32) || (SrcFP && Dst.First < 32);
  if (((Dst.First < 32) != (Src->First < 32) || DstFP != SrcFP) &&
      !(IntegerFPCopy && Dst.Size == 1 && FPRegBytes == ST.getXLen() / 8))
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
  if (&MBB == &MF.front() || MBB.isEHPad() || MBB.hasAddressTaken()) {
    Incoming.emplace(Registers.size());
    // The standard ABI requires vstart=0 at function entry. Interrupt and
    // naked functions do not provide this guarantee.
    Incoming->VStartIsZero = &MBB == &MF.front() &&
                             !MF.getFunction().hasFnAttribute("interrupt") &&
                             !MF.getFunction().hasFnAttribute(Attribute::Naked);
  }
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

bool RISCVRedundantInstrElimImpl::getVStartIsZero(
    const MachineInstr &MI, const AvailableValues &Values) const {
  if (MI.isCall() || MI.isInlineAsm() || MI.isBundled())
    return false;
  unsigned Opcode = MI.getOpcode();
  if (isCSRInstruction(Opcode)) {
    if (!MI.getOperand(1).isImm() || MI.getOperand(1).getImm() != 0x008)
      return Values.VStartIsZero;
    auto Src = getConstant(MI.getOperand(2), Values);
    switch (Opcode) {
    default:
      llvm_unreachable("Unexpected CSR instruction");
    case RISCV::CSRRW:
    case RISCV::CSRRWI:
      return Src && *Src == 0;
    case RISCV::CSRRS:
    case RISCV::CSRRSI:
      return Values.VStartIsZero && Src && *Src == 0;
    case RISCV::CSRRC:
    case RISCV::CSRRCI:
      return Values.VStartIsZero || (Src && *Src == -1);
    }
  }
  // vset{i}vl{i} resets vstart. Check before the side-effect test, since the MC
  // forms are marked as having unmodeled side effects.
  switch (Opcode) {
  default:
    break;
  case RISCV::VSETVLI:
  case RISCV::VSETIVLI:
  case RISCV::VSETVL:
  case RISCV::PseudoVSETVLI:
  case RISCV::PseudoVSETIVLI:
  case RISCV::PseudoVSETVLIX0:
  case RISCV::PseudoVSETVLIX0X0:
    return true;
  }
  if (MI.hasUnmodeledSideEffects())
    return false;
  if (getVectorCopySize(Opcode) || RISCV::getRVVMCOpcode(Opcode))
    return true;
  switch (Opcode) {
  default:
    return Values.VStartIsZero;
#define WHOLE_VECTOR_MEMORY(N)                                                 \
  case RISCV::VL##N##RE8_V:                                                    \
  case RISCV::VL##N##RE16_V:                                                   \
  case RISCV::VL##N##RE32_V:                                                   \
  case RISCV::VL##N##RE64_V:                                                   \
  case RISCV::VS##N##R_V:
    WHOLE_VECTOR_MEMORY(1)
    WHOLE_VECTOR_MEMORY(2)
    WHOLE_VECTOR_MEMORY(4)
    WHOLE_VECTOR_MEMORY(8)
#undef WHOLE_VECTOR_MEMORY
    return true;
  }
}

void RISCVRedundantInstrElimImpl::processInstruction(
    MachineInstr &MI, AvailableValues &Values,
    SmallVectorImpl<MachineInstr *> *Redundant) {
  if (MI.isDebugInstr())
    return;
  // vstart only matters for whole-register moves.
  bool VStartIsZero = VectorRegBase && getVStartIsZero(MI, Values);
  auto FI = getSpillSlot(MI);
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
        auto Src = getCopySource(MI, *Dst, Values);
        if (Src) {
          for (unsigned I = 0; I != Src->Size; ++I)
            Result.push_back(Values.get(Src->First + I));
        } else if (FI && MI.mayLoad() && Dst->Size == 1) {
          Result.push_back(Values.getSpill(*FI));
        }
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
    if (MI.mayLoad())
      ++NumReloadsRemoved;
    else if (Constant)
      ++NumConstantsRemoved;
    else
      ++NumCopiesRemoved;
    return;
  }

  // Snapshot store operands before invalidating any register or alias.
  std::optional<ValueInfo> Stored;
  if (ModelValue && FI && MI.mayStore()) {
    auto Src = getRegisterRange(MI.getOperand(0));
    if (Src && Src->Size == 1)
      Stored = Values.get(Src->First);
  }
  bool HasRegMask = llvm::any_of(
      MI.operands(), [](const MachineOperand &MO) { return MO.isRegMask(); });
  // Calls and inline assembly describe their register effects with operands.
  // Other unmodeled effects and calls without a mask are opaque barriers.
  bool Opaque = MI.hasUnmodeledSideEffects() && !MI.isCall() &&
                !MI.isInlineAsm() && !isCSRInstruction(MI.getOpcode());
  bool ClobberAll = Opaque || MI.isBundled() || (MI.isCall() && !HasRegMask);
  bool ClobbersStackBase = ClobberAll;
  auto Clobber = [&](unsigned I) {
    Values.forget(I);
    ClobbersStackBase |= llvm::is_contained(StackBaseLocations, I);
  };
  if (ClobberAll) {
    for (unsigned I = 1, E = Registers.size(); I != E; ++I)
      Values.forget(I);
  } else {
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isRegMask()) {
        for (unsigned I = 1, E = Registers.size(); I != E; ++I)
          if (MO.clobbersPhysReg(Registers[I]))
            Clobber(I);
      } else if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical()) {
        for (MCRegUnit Unit : TRI.regunits(MO.getReg()))
          if (unsigned I = RegUnitLocations[static_cast<unsigned>(Unit)])
            Clobber(I);
      }
    }
  }

  // Different frame indices can overlap after stack coloring/layout. Any store
  // invalidates all spill facts; remember only an exact full-slot store.
  if (MI.mayStore() || MI.hasOrderedMemoryRef() || MI.isCall() ||
      MI.isInlineAsm() || Opaque || MI.isBundled() || ClobbersStackBase)
    Values.forgetSpills();
  if (Stored && !ClobbersStackBase)
    Values.assignSpill(*FI, *Stored);
  if (Constant)
    Values.assignConstant(Dst->First, *Constant);
  for (auto [I, Value] : llvm::enumerate(Result))
    Values.assign(Dst->First + I, Value);
  Values.VStartIsZero = VStartIsZero;
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
