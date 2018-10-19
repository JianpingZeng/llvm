//
// Created by xlous on 9/12/18.
//

#include <deque>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include "IdemUtil.h"

using namespace std;
using namespace llvm;

#ifndef NDEBUG
bool llvm::DebugIdemFlag;

// -idem-debug - Command line option to enable the DEBUG statements in the idem passes.
// This flag may only be enabled in debug builds.
static cl::opt<bool, true>
    IdemDebug("idem-debug", cl::desc("Enable debug output on Idempotence"), cl::Hidden,
          cl::location(DebugIdemFlag));

#endif

static void computeReversePostOrder(MachineBasicBlock *root,
                                    std::set<MachineBasicBlock*> &visited,
                                    std::vector<MachineBasicBlock *> &sequence) {
  if (!root) return;
  if (!visited.insert(root).second) return;

  if (!root->succ_empty()) {
    for (auto succ = root->succ_rbegin(), end = root->succ_rend(); succ != end; ++succ)
      computeReversePostOrder(*succ, visited, sequence);
  }
  sequence.push_back(root);
}

void llvm::computeReversePostOrder(MachineFunction &MF,
                             MachineDominatorTree &dt,
                             std::vector<MachineBasicBlock *> &sequence) {
  sequence.clear();
  if (MF.empty())
    return;

  std::set<MachineBasicBlock*> visited;
  ::computeReversePostOrder(&MF.front(), visited, sequence);
  std::reverse(sequence.begin(), sequence.end());
}

bool llvm::reachable(MachineInstr *A, MachineInstr *B) {
  if (!A || !B) return false;

  if (!A->getParent() || !B->getParent())
    return false;

  if (A == B) return true;

  if (A->getParent() == B->getParent()) {
    for (MachineInstr &mi : *A->getParent()) {
      if (&mi == A)
        return true;
      else if (&mi == B)
        return false;
    }
    assert(false);
  }

  if (A->getParent() != B->getParent()) {
    MachineBasicBlock *MBBA = A->getParent();
    MachineBasicBlock *MBBB = B->getParent();

    std::vector<MachineBasicBlock*> worklist;
    worklist.push_back(MBBA);

    std::set<MachineBasicBlock*> visited;

    while (!worklist.empty()) {
      auto cur = worklist.back();
      worklist.pop_back();
      if (!visited.count(cur))
        continue;

      if (cur == MBBB)
        return true;

      std::for_each(cur->succ_begin(), cur->succ_end(), [&](MachineBasicBlock *mbb)
            { return worklist.push_back(mbb); });
    }
  }
  return false;
}

char llvm::IdemInstrScavenger::ID = 0;

