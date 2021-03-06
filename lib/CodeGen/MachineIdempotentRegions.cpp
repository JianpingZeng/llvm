//===-------- MachineIdempotentRegions.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for querying and updating the
// idempotent region information at the machine level.  A "machine" idempotent
// region is defined by the single IDEM instruction that defines its entry point
// and it spans all instructions reachable by control flow from the entry point
// to subsequent IDEM instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "machine-idempotent-regions"
#include "llvm/CodeGen/MachineIdempotentRegions.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/IdempotenceOptions.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// IdempotentRegion
//===----------------------------------------------------------------------===//

void IdempotentRegion::dump() const {
  print(dbgs());
}

void IdempotentRegion::print(raw_ostream &OS, const SlotIndexes *SI) const {
  OS << "IR#" << ID_ << " ";
  if (SI)
    OS << "@" << SI->getInstructionIndex(&getEntry()) << " ";
  OS << "in BB#" << getEntryMBB().getNumber();
}

raw_ostream &llvm::operator<<(raw_ostream &OS, const IdempotentRegion &R) {
  R.print(OS);
  return OS;
}

//===----------------------------------------------------------------------===//
// MachineIdempotentRegions
//===----------------------------------------------------------------------===//

char MachineIdempotentRegions::ID = 0;
INITIALIZE_PASS(MachineIdempotentRegions,
                "machine-idempotence-regions",
                "Machine Idempotent Regions", false, true)

FunctionPass *llvm::createMachineIdempotentRegionsPass() {
  return new MachineIdempotentRegions();
}

void MachineIdempotentRegions::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void MachineIdempotentRegions::releaseMemory() {
  RegionAllocator_.Reset();
  Regions_.clear();
  EntryToRegionMap_.clear();
}

bool MachineIdempotentRegions::runOnMachineFunction(MachineFunction &MF) {
  assert((IdempotenceConstructionMode != IdempotenceOptions::NoConstruction ||
      EnableRegisterRenaming) && "pass should not be run");

  MF_  = &MF;
  TII_ = MF.getTarget().getInstrInfo();
  TRI_ = MF.getTarget().getRegisterInfo();

  // Regions start at idem boundaries.
  for (MachineFunction::iterator B = MF.begin(), BE = MF.end(); B != BE; ++B)
    for (MachineBasicBlock::iterator I = B->begin(); I != B->end(); ++I)
      if (TII_->isIdemBoundary(I))
        createRegionAtBoundary(I);

  return false;
}

IdempotentRegion &MachineIdempotentRegions::createRegionAtBoundary(
    MachineInstr *MI) {
  assert(isRegionEntry(*MI) && "creating region at non-boundary");

  IdempotentRegion *Region =
    new (RegionAllocator_) IdempotentRegion(Regions_.size(), MI);
  Regions_.push_back(Region);
  assert(EntryToRegionMap_.insert(std::make_pair(MI, Region)).second &&
         "already in map");
  return *Region;
}

IdempotentRegion &MachineIdempotentRegions::createRegionBefore(
    MachineBasicBlock *MBB,
    MachineBasicBlock::iterator MI,
    SlotIndexes *Indexes) {

  // The new region starts at I.
  TII_->emitIdemBoundary(*MBB, MI);

  // Update Indexes as needed.
  MachineBasicBlock::iterator Boundary = prior(MI);
  if (Indexes)
    Indexes->insertMachineInstrInMaps(Boundary);

  return createRegionAtBoundary(Boundary);
}

void MachineIdempotentRegions::getRegionsContaining(
  const MachineInstr &MI,
  std::vector<IdempotentRegion *> *Regions) {

  // Clear the return argument.
  Regions->clear();

  // Walk the CFG backwards, starting at the instruction before MI.
  typedef MachineBasicBlock::const_reverse_iterator ITR;
  struct WorkItemTy {
    ITR begin;
    ITR end;
    const MachineBasicBlock *mbb;
    WorkItemTy(const ITR Begin, const ITR End, const MachineBasicBlock *MBB) : begin(Begin), end(End),
                                                             mbb(MBB) {}
  };

  std::vector<WorkItemTy> Worklist;
  Worklist.emplace_back(ITR(&MI), MI.getParent()->rend(), MI.getParent());

  std::set<const MachineBasicBlock *> Visited;
  do {
    ITR It, end;
    It = Worklist.back().begin;
    end = Worklist.back().end;
    const MachineBasicBlock *mbb = Worklist.back().mbb;
    Worklist.pop_back();

    if (!Visited.insert(mbb).second)
      continue;

    if (It != end) {
      // Look for a region entry or the block entry, whichever comes first.
      while (It != end && !isRegionEntry(*It))
        It++;

      // If we found a region entry, add the region and skip predecessors.
      if (It != end) {
        Regions->push_back(&getRegionAtEntry(*It));
        continue;
      }
    }

    // Examine predecessors.  Insert into Visited here to allow for a cycle back
    // to MI's block.
    for (auto P = mbb->pred_begin(), PE = mbb->pred_end(); P != PE; ++P)
      Worklist.emplace_back((*P)->rbegin(), (*P)->rend(), *P);

  } while (!Worklist.empty());
}

#if 0
static void dumpVerifying(const MachineInstr &MI,
                          const DenseSet<unsigned> &LiveIns,
                          const SlotIndexes *Indexes,
                          const TargetRegisterInfo *TRI) {
  dbgs() << "For live-ins: [";
  for (DenseSet<unsigned>::const_iterator I = LiveIns.begin(),
       IE = LiveIns.end(), First = I; I != IE; ++I) {
    if (I != First)
      dbgs() << ", ";
    dbgs() << PrintReg(*I, TRI);
  }
  dbgs() << "], verifying instruction: ";
  if (Indexes)
    dbgs() << "\t" << Indexes->getInstructionIndex(&MI);
  dbgs() << "\t\t" << MI;
}
#endif

bool MachineIdempotentRegions::verifyInstruction(
    const MachineInstr &MI,
    const DenseSet<unsigned> &LiveIns,
    const SlotIndexes *Indexes) const {

  // Identity copies and kills don't really write to anything.
  if (MI.isIdentityCopy() || MI.isKill())
    return true;

  bool Verified = true;
  for (MachineInstr::const_mop_iterator O = MI.operands_begin(),
       OE = MI.operands_end(); O != OE; ++O)
    Verified &= verifyOperand(*O, LiveIns, Indexes);
  return Verified;
}

bool MachineIdempotentRegions::verifyOperand(
    const MachineOperand &MO,
    const DenseSet<unsigned> &LiveIns,
    const SlotIndexes *Indexes) const {
  unsigned Reg = 0;

  // For registers, consider only defs ignoring:
  //  - Undef defs, which are generated while RegisterCoalescer is running.
  //  - Implicit call defs.  They are handled by an idempotence boundary at the
  //    entry of the called function.
  if (MO.isReg() && MO.isDef() &&
      !(MO.isUndef() && MO.getParent()->isCopyLike()) &&
      !(MO.isImplicit() && MO.getParent()->isCall())) {
    Reg = MO.getReg();
    // Alse ignore:
    //  - Stack pointer defs; assume the SP is checkpointed at idempotence
    //    boundaries.
    //  - Condition code defs; assume the CCR is checkpointed at idempotence
    //    boundaries.  The SelectionDAG scheduler currently allows a CCR to be
    //    live across a boundary (could fix that instead).
    //  - Other target-specific special registers that are hard to handle.
    if (TargetRegisterInfo::isPhysicalRegister(Reg) &&
        TRI_->isProtectedRegister(Reg))
      return true;
  }

  // For frame indicies, consider only spills (stores, index > 0) for now.
  if (MO.isFI() && MO.getParent()->mayStore() && MO.getIndex() > 0)
    Reg = TargetRegisterInfo::index2StackSlot(MO.getIndex());

  // If Reg didn't get set, assume everything is fine.
  if (!Reg)
    return true;

  bool Verified = !LiveIns.count(Reg);
  if (!Verified) {
    errs() << PrintReg(Reg, TRI_) << " CLOBBER in:";
    /*if (Indexes)
      errs() << "\t" << Indexes->getInstructionIndex(MO.getParent());*/
    errs() << "\t\t" << *MO.getParent();

    /*llvm::errs()<<"LiveIns: [";
    for (auto &r : LiveIns)
      llvm::errs()<<PrintReg(r, TRI_)<<",";
    llvm::errs()<<"]\n";*/
  }
  return Verified;
}

void MachineIdempotentRegions::print(raw_ostream &OS, const Module *) const {
  OS << "\n*** MachineIdempotentRegions: ***\n";
  for (const_iterator R = begin(), RE = end(); R != RE; ++R) {
    OS << **R << "\n";
  }
}


