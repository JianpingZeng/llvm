#include <llvm/ADT/SetOperations.h>
#include "IdempotentRegionLiveInsGather.h"

using namespace llvm;

void LiveInsGather::run() {

  // Compute local live set.
  for (auto &mbb : mf) {
    liveGens[&mbb] = RegSet();
    liveKills[&mbb] = RegSet();

    for (auto mi = mbb.rbegin(), end = mbb.rend(); mi != end; ++mi) {
      for (int j = mi->getNumOperands()-1; j >= 0; j--) {
        auto mo = mi->getOperand(j);
        if (!mo.isReg() || !mo.getReg()) continue;
        int reg = mo.getReg();
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
      auto itr = lhs.begin(), end = rhs.end();
      auto rhsItr = rhs.begin();
      for (; itr != end; ++itr, ++rhsItr) {
        if (*itr != *rhsItr)
          return false;
      }
      return rhsItr == rhs.end();
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

      set_union(out, liveOutMBBMap[&*mbb]);
      changed = !RegSetEq(out, liveOutMBBMap[&*mbb]);
      if (changed)
        liveOutMBBMap[&*mbb] = out;

      auto in = liveOutMBBMap[&*mbb];
      set_intersect(in, liveKills[&*mbb]);
      set_union(in, liveGens[&*mbb]);
      changed = !RegSetEq(in, liveInMBBMap[&*mbb]);
      if (changed)
        liveInMBBMap[&*mbb] = in;
    }
  } while (changed);


  for (auto &mbb : mf) {
    for (auto &mi : mbb) {
      if (tii->isIdemBoundary(&mi)) {
        computeIdemLiveIns(&mi);
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
      if (!mo.isReg() || !mo.getReg())
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