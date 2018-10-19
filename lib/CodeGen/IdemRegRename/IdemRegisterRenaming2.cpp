#include <utility>

//===----- IdemRegisterRenaming.cpp - Register regnaming after RA ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "reg-renaming"

#include <llvm/PassSupport.h>
#include <llvm/CodeGen/MachineIdempotentRegions.h>
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "LiveIntervalAnalysisIdem.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "IdemUtil.h"
#include "IdempotentRegionLiveInsGather.h"

using namespace llvm;

/// @author Jianping Zeng.
namespace {

struct AntiDeps {
  unsigned reg;
  std::vector<MachineOperand *> uses;
  std::vector<MachineOperand *> defs;
};

class IdemRegisterRenamer : public MachineFunctionPass {
public:
  static char ID;
  IdemRegisterRenamer() : MachineFunctionPass(ID) {
    //initializeRegisterRenamingPass(*PassRegistry::getPassRegistry());
  }

  virtual bool runOnMachineFunction(MachineFunction &MF) override;

  virtual void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalAnalysisIdem>();
    AU.addRequired<MachineIdempotentRegions>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  const char *getPassName() const {
    return "Register Renaming for Idempotence pass";
  }
  virtual void releaseMemory() {
  }

private:
  inline void collectLiveInRegistersForRegions();
  void computeAntiDependenceSet();
  void gatherAntiDeps(MachineInstr *idem);
  bool handleAntiDependences();
  void collectAntiDepsTrace(unsigned,
                            MachineBasicBlock::iterator,
                            MachineBasicBlock::iterator,
                            MachineBasicBlock *,
                            std::set<MachineBasicBlock *> &,
                            std::vector<MachineOperand *>,
                            std::vector<MachineOperand *>);
  bool isTwoAddressInstr(MachineInstr *useMI);

private:
  const TargetInstrInfo *tii;
  MachineIdempotentRegions *mir;
  std::vector<MachineBasicBlock *> reversePostOrderMBBs;
  std::vector<AntiDeps> antiDeps;
  LiveInsGather *gather;
};
}

INITIALIZE_PASS_BEGIN(IdemRegisterRenamer, "reg-renaming",
                      "Register Renaming for Idempotence", false, false)
  INITIALIZE_PASS_DEPENDENCY(LiveIntervalAnalysisIdem)
  INITIALIZE_PASS_DEPENDENCY(MachineIdempotentRegions)
INITIALIZE_PASS_END(IdemRegisterRenamer, "reg-renaming",
                    "Register Renaming for Idempotence", false, false)

char IdemRegisterRenamer::ID = 0;

void IdemRegisterRenamer::collectLiveInRegistersForRegions() {
  gather->run();
}

static bool contains(std::vector<MachineOperand *> &set, int reg) {

}

void IdemRegisterRenamer::collectAntiDepsTrace(unsigned reg,
                                               MachineBasicBlock::iterator idem,
                                               MachineBasicBlock::iterator end,
                                               MachineBasicBlock *mbb,
                                               std::set<MachineBasicBlock *> &visited,
                                               std::vector<MachineOperand *> uses,
                                               std::vector<MachineOperand *> defs) {
  if (!visited.insert(mbb).second)
    return;

  for (auto itr = idem; itr != end; ++itr) {
    if (tii->isIdemBoundary(itr))
      return;

    for (int i = 0, e = itr->getNumOperands(); i < e; i++) {
      auto mo = itr->getOperand(i);
      if (!mo.isReg() || !mo.getReg())
        continue;

      if (mo.isUse()) {
        if (contains(defs, mo.getReg()))
          goto CONST_IDEM;

        uses.push_back(&mo);
      }
      else
        defs.push_back(&mo);
    }
  }

  if (mbb && !mbb->succ_empty()) {
    for (auto succ = mbb->succ_begin(), succEnd = mbb->succ_end(); succ != succEnd; ++succ) {
      collectAntiDepsTrace(reg, (*succ)->begin(), (*succ)->end(), *succ, visited, uses, defs);
    }
  }

  // Construct anti-dependences accoridng uses and defs set.
CONST_IDEM:
  antiDeps.push_back({reg, uses, defs});
  return;
}

void IdemRegisterRenamer::gatherAntiDeps(MachineInstr *idem) {
  auto liveIns = gather->getIdemLiveIns(idem);
  if (liveIns.empty())
    return;

  std::set<MachineBasicBlock*> visited;
  for (auto reg : liveIns) {
    collectAntiDepsTrace(reg, MachineBasicBlock::iterator(idem),
                         idem->getParent()->end(), idem->getParent(),
                         visited, std::vector<MachineOperand *>(),
                         std::vector<MachineOperand *>());
  }
}

void IdemRegisterRenamer::computeAntiDependenceSet() {
  for (auto itr = mir->begin(), end = mir->end(); itr != end; ++itr) {
    MachineInstr *idem = &(*itr)->getEntry();
    assert(idem && tii->isIdemBoundary(idem));
    gatherAntiDeps(idem);
  }
}

bool IdemRegisterRenamer::isTwoAddressInstr(MachineInstr *useMI) {
  // We should not rename the two-address instruction.
  auto MCID = useMI->getDesc();
  int numOps = useMI->isInlineAsm() ? useMI->getNumOperands() : MCID.getNumOperands();
  for (int i = 0; i < numOps; i++) {
    unsigned destIdx;
    if (!useMI->isRegTiedToDefOperand(i, &destIdx))
      continue;

    return true;
  }
  return false;
}

bool IdemRegisterRenamer::handleAntiDependences() {
  if (antiDeps.empty())
    return false;

  for (auto &pair : antiDeps) {
    if (pair.uses.size() == 1) {
      if (isTwoAddressInstr(pair.uses[0]->getParent())) {
        
      }
      else {

      }
    }


    // get the last insertion position of previous adjacent region
    // or the position of prior instruction depends on if the current instr
    // is a two address instr.

    // get the free register
    //
  }

  return true;
}

bool IdemRegisterRenamer::runOnMachineFunction(MachineFunction &MF) {
  // Step#1: Compute the live-in registers set for each idempotence region.
  // Step#2: Handle two address instruction, insert a move instr before it right away.
  // Step#3: Determine whether I should insert a move instr for anti-dependence or
  //         replace the register name of anti-dependence.
  //
  //         When there are enough free registers, we take the method that
  //         replacing the register name rather than inserting a move instr in
  //         in the previous region(when no such region, create a new region before current region)
  gather = new LiveInsGather(MF);
  mir = getAnalysisIfAvailable<MachineIdempotentRegions>();
  assert(mir && "No MachineIdempotentRegions available!");
  tii = MF.getTarget().getInstrInfo();

  // Collects anti-dependences operand pair.
  collectLiveInRegistersForRegions();
  computeReversePostOrder(MF, reversePostOrderMBBs);

  computeAntiDependenceSet();

  bool changed = false;
  changed |= handleAntiDependences();
  delete gather;
  return changed;
}