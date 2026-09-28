#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <fstream>
#include <string>
#include <iomanip>
void exportOne(const char* species,const char* epoch){
 std::string base="/home/comet/davidz/ReconDev/";
 std::string in=base+"ReconData/runs/ecal_clustering/"+species+"_20to120MeV_10cmPlateGun_0To10deg/step1/outputs/eval_test_epoch_"+epoch+".root";
 TFile f(in.c_str());TTree* tree=(TTree*)f.Get("events");TTreeReader r(tree);
 TTreeReaderArray<float> truth(r,"truth_t0"),pred(r,"pred_oracle_t0"),reco(r,"reco_t0"),tmin(r,"window_tmin_ns"),wt(r,"window_truth_t0_ns"),energy(r,"truth_ecluster");
 TTreeReaderArray<Long64_t> w(r,"truth_prompt_window"),match(r,"truth_match_reco_index"),anchor(r,"reco_anchor_window"),src(r,"association_src"),dst(r,"association_dst"),assoc(r,"association_target"),role(r,"window_role_target");
 TTreeReaderValue<Int_t> file(r,"file_index");TTreeReaderValue<Long64_t> event(r,"event"),graph(r,"graph_index_in_file");
 std::ofstream out(base+"ecal_clustering_gnn_trainer/diagnostics/particle_timing_comparison/"+species+".csv");out<<std::setprecision(12)<<"file,event,graph,object,truth_t0,tmin,oracle_t0,reco_t0,window_truth_t0,matched,same_anchor,has_delayed,nwindows,energy\n";
 while(r.Next())for(size_t i=0;i<truth.GetSize();i++){
  int wi=w[i],m=match[i];if(wi<0||wi>=tmin.GetSize())continue;bool delayed=false;
  for(size_t j=0;j<assoc.GetSize();j++){if(!assoc[j])continue;int other=src[j]==wi?dst[j]:(dst[j]==wi?src[j]:-1);if(other>=0&&other<role.GetSize()&&role[other]==2)delayed=true;}
  out<<*file<<","<<*event<<","<<*graph<<","<<i<<","<<truth[i]<<","<<tmin[wi]<<","<<pred[i]<<",";if(m>=0)out<<reco[m];else out<<"nan";
  out<<","<<wt[wi]<<","<<(m>=0)<<","<<(m>=0&&anchor[m]==wi)<<","<<delayed<<","<<tmin.GetSize()<<","<<energy[i]<<"\n";
 }
}
void export_timing(){exportOne("mum","048");exportOne("em","049");exportOne("pim","049");}
