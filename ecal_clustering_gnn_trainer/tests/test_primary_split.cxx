#include "../reconstruct_ecal_clusters_from_eval_root.cxx"
#include <cassert>
#include <cstdio>
int main() {
  gROOT->SetBatch(kTRUE);
  assert(Category(-11)==1 && Category(13)==2 && Category(-211)==3 && Category(22)==0);
  Split1D split("split","",10,0,10);
  for(int category : {0,1,2,3,1}) { primaryCategory=category; split.Fill(2); }
  assert(split.GetEntries()==5 && split.species[0]->GetEntries()==2);
  assert(split.species[1]->GetEntries()==1 && split.species[2]->GetEntries()==1);
}
