#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <map>
#include <array>
#include <iostream>
#include <cmath>
void onePID(const char* s,const char* ep){
 std::string p="/home/comet/davidz/ReconDev/ReconData/runs/ecal_clustering/"+std::string(s)+"_20to120MeV_10cmPlateGun_0To10deg/step1/outputs/eval_test_epoch_"+ep+".root";
 TFile f(p.c_str());TTreeReader r((TTree*)f.Get("events"));TTreeReaderArray<Long64_t> pid(r,"truth_pid"),w(r,"truth_prompt_window"),match(r,"truth_match_reco_index");TTreeReaderArray<float> truth(r,"truth_t0"),tmin(r,"window_tmin_ns"),pred(r,"pred_oracle_t0");std::map<int,std::array<long,5>> counts;
 while(r.Next())for(size_t i=0;i<pid.GetSize();i++){double dt=tmin[w[i]]-truth[i];double corr=tmin[w[i]]-pred[i];auto &c=counts[pid[i]];c[0]++;c[1]+=std::abs(dt)<=1;c[2]+=dt>30;c[3]+=std::abs(corr+1.065601)<.001;c[4]+=match[i]>=0;}
 std::cout<<s<<" pid total small_dt late_dt constant_correction matched\n";for(auto &kv:counts){std::cout<<kv.first;for(auto n:kv.second)std::cout<<" "<<n;std::cout<<"\n";}
}
void check_target_pid(){onePID("mum","048");onePID("pim","049");}
