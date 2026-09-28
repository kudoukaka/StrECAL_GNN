#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <cmath>
#include <iostream>
#include <iomanip>
struct S { double n=0,s=0,ss=0; void add(double x){n++;s+=x;ss+=x*x;} double mean(){return s/n;} double sd(){return sqrt(ss/n-mean()*mean());}};
void one(const char* setup,const char* epoch){
 std::string path="/home/comet/davidz/ReconDev/ReconData/runs/ecal_clustering/"+std::string(setup)+"/step1/outputs/eval_test_epoch_"+epoch+".root";
 TFile f(path.c_str()); TTree* t=(TTree*)f.Get("events"); TTreeReader r(t);
 TTreeReaderArray<float> y(r,"truth_y"),z(r,"truth_z"),cy(r,"window_centroid_y_mm"),cz(r,"window_centroid_z_mm"),py(r,"pred_oracle_y"),pz(r,"pred_oracle_z"),ry(r,"reco_y"),rz(r,"reco_z");
 TTreeReaderArray<Long64_t> w(r,"truth_prompt_window"),match(r,"truth_match_reco_index");
 S base[2],corr[2],res[2],matched[2];double product[2]={0,0};
 while(r.Next()){for(size_t i=0;i<y.GetSize();i++){if(w[i]<0||w[i]>=cy.GetSize()) continue;double b[2]={cy[w[i]]-y[i],cz[w[i]]-z[i]},c[2]={cy[w[i]]-py[i],cz[w[i]]-pz[i]};for(int a=0;a<2;a++){base[a].add(b[a]);corr[a].add(c[a]);res[a].add(b[a]-c[a]);product[a]+=b[a]*c[a];} if(match[i]>=0){matched[0].add(ry[match[i]]-y[i]);matched[1].add(rz[match[i]]-z[i]);}}}
 std::cout<<std::setprecision(10)<<"RESULT "<<setup<<" "<<epoch<<"\n";
 for(int a=0;a<2;a++)std::cout<<(a==0?"Y":"Z")<<" N "<<res[a].n<<" baseline_sd "<<base[a].sd()<<" correction_mean "<<corr[a].mean()<<" correction_sd "<<corr[a].sd()<<" correction_target_corr "<<(product[a]/base[a].n-base[a].mean()*corr[a].mean())/(base[a].sd()*corr[a].sd())<<" oracle_sd "<<res[a].sd()<<" matched_N "<<matched[a].n<<" matched_sd "<<matched[a].sd()<<" matched_bias "<<matched[a].mean()<<"\n";
}
void compare_electron(){one("em_regression_20to120MeV_10cmPlateGun_0To10deg","036");for(auto e:{"025","035","046","049"})one("em_20to120MeV_10cmPlateGun_0To10deg",e);}
