#include <llvm/ADT/SetOperations.h>
#include <llvm/Support/raw_ostream.h>
#include "IdempotentRegionLiveInsGather.h"

using namespace llvm;

void LiveInsGather::run() {

  // Compute local live set.
  for (auto &mbb : mf) {
    liveGens[&mbb] = RegSet();
    liveKills[&mbb] = RegSet();

    for (auto mi = mbb.begin(), end = mbb.end(); mi != end; ++mi) {
      for (int j = mi->getNumOperands() - 1; j >= 0; j--) {
        auto mo = mi->getOperand(j);

        if (!mo.isReg() || !mo.getReg() ||
            // We don't count such special registers
            tri->isNotCountedAsLiveness(mo.getReg())) continue;
        unsigned reg = mo.getReg();
        if (mo.isUse() && !liveKills[&mbb].count(reg))
          liveGens[&mbb].insert(reg);
        else
          liveKills[&mbb].insert(reg);
      }
    }
  }

  // Compute global live set.
  for (auto &mbb : mf) {
    liveInMBBMap[&mbb] = RegSet();
    liveOutMBBMap[&mbb] = RegSet();
  }

  auto RegSetEq = [](RegSet lhs, RegSet rhs) {
      if (lhs.size() != rhs.size()) return false;
      auto itr = lhs.begin(), end = lhs.end();
      auto rhsItr = rhs.begin();
      for (; itr != end; ++itr, ++rhsItr) {
        if (*itr != *rhsItr)
          return false;
      }
      return true;
  };

  bool changed;
  do {
    changed = false;
    for (auto mbb = mf.rbegin(), end = mf.rend(); mbb != end; ++mbb) {
      auto out = RegSet();
      if (!mbb->succ_empty()) {
        for (auto succ = mbb->succ_begin(), succEnd = mbb->succ_end(); succ != succEnd; ++succ)
          set_union(out, liveInMBBMap[*succ]);
      }

      /*if (mbb->getName() == "if.end35") {
        llvm::errs()<<"Before: ";
        printLiveRegisters(liveOutMBBMap[&*mbb], false);
      }*/

      set_union(out, liveOutMBBMap[&*mbb]);
      bool localChanged = !RegSetEq(out, liveOutMBBMap[&*mbb]);
      if (localChanged) {
        changed = true;
        liveOutMBBMap[&*mbb] = out;
      }

      /*if (mbb->getName() == "if.end35") {
        llvm::errs()<<"After: ";
        printLiveRegisters(liveOutMBBMap[&*mbb], false);
      }*/

      auto in = out;
      set_subtract(in, liveKills[&*mbb]);
      set_union(in, liveGens[&*mbb]);
      localChanged = !RegSetEq(in, liveInMBBMap[&*mbb]);
      if (localChanged) {
        changed = true;
        liveInMBBMap[&*mbb] = in;
      }
    }
  } while (changed);

  // Print out live-in registers set for each machine basic block.
  for (auto &mbb : mf) {
    /*if (mbb.getName() == "if.end35") {
      printLiveRegisters(liveOutMBBMap[&mbb], false);
    }
    if (mbb.getName() == "for.body")
      printLiveRegisters(liveInMBBMap[&mbb], true);*/

    /*llvm::errs()<<mbb.getName()<<", ";
    printLiveRegisters(liveInMBBMap[&mbb]);
    printLiveRegisters(liveOutMBBMap[&mbb], false);*/
    for (auto &mi : mbb) {
      if (tii->isIdemBoundary(&mi)) {
        computeIdemLiveIns(&mi);
        /*if (mbb.getName() == "if.end35") {
          llvm::errs()<<"Idem, ";
          printLiveRegisters(idemLiveInMap[&mi]);
        }*/
      }
    }
  }
}

void LiveInsGather::computeIdemLiveIns(const MachineInstr *mi) {
  if (!mi || !tii->isIdemBoundary(mi)) return;
  auto mbb = mi->getParent();
  assert(mbb);

  auto liveOuts = liveOutMBBMap[mbb];
  auto end = MachineBasicBlock::const_reverse_iterator(mi);
  for (auto itr = mbb->rbegin(); itr != end; ++itr) {
    for (int i = 0, e = itr->getNumOperands(); i < e; i++) {
      auto mo = itr->getOperand(i);

      if (!mo.isReg() || !mo.getReg() ||
          // We don't count such special registers
          tri->isNotCountedAsLiveness(mo.getReg()))
        continue;

      if (mo.isDef())
        liveOuts.erase(mo.getReg());
      else
        liveOuts.insert(mo.getReg());
    }
  }

  // assign the live out to the idem's live in
  idemLiveInMap[mi] = liveOuts;
}

void LiveInsGather::printLiveRegisters(llvm::LiveInsGather::RegSet &regs, bool liveInOrLiveOut) {
  size_t i = 0, e = regs.size();
  if (liveInOrLiveOut)
    llvm::errs()<<"LiveIns: [";
  else
    llvm::errs()<<"LiveOuts: [";
  for (auto reg :  regs) {
    llvm::errs() << tri->getName(reg);
    if (i < e - 1)
      llvm::errs()<<",";
    ++i;
  }

  llvm::errs()<<"]\n";
}