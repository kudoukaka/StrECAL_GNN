// Compile with ROOT flags and -std=c++14, then run from any directory.
#include "../reconstruct_ecal_clusters_from_eval_root.cxx"
#include <cassert>
#include <cstdio>

namespace {
void Near(double a, double b) { assert(std::abs(a-b) < 1e-9 * std::max(1.0,std::abs(b))); }
}

int main() {
  ResidualHistogram biased("biased","",10,-5,5);
  biased.Fill(2); biased.Fill(4);
  Near(biased.Mean(),3); Near(biased.StdDev(),1); Near(biased.RMSE(),std::sqrt(10.0));

  ResidualHistogram full("full","",10,-5,5);
  for (double x : {-100.,2.,4.,100.}) full.Fill(x);
  full.Fill(std::numeric_limits<double>::quiet_NaN());
  full.Fill(std::numeric_limits<double>::infinity());
  assert(full.count==4 && full.invalid==2 && full.underflow==1 && full.overflow==1);
  Near(full.Mean(),1.5); Near(full.RMSE(),std::sqrt(5005.));
  Near(full.StdDev(),std::sqrt(5002.75));
  Near(full.GetMean(),full.Mean()); Near(full.GetStdDev(),full.StdDev());
  full.GetXaxis()->SetRange(1,10);
  Near(full.Mean(),1.5); Near(full.RMSE(),std::sqrt(5005.));
  full.GetXaxis()->SetRange(0,0);

  ResidualHistogram empty("empty","",2,-1,1);
  assert(std::isnan(empty.Mean()) && std::isnan(empty.StdDev()) && std::isnan(empty.RMSE()));
  ResidualHistogram boundary("boundary","",2,-1,1);
  boundary.Fill(-1); boundary.Fill(1);
  assert(boundary.underflow==0 && boundary.overflow==1);
  ResidualSummary radial("radial","test");
  radial.Fill(0,-3,4); radial.Fill(100,3,-4);
  Near(radial.yzRadial->Mean(),5); Near(radial.yzRadial->RMSE(),5);
  Near(radial.t0->Mean(),50); Near(radial.t0->StdDev(),50);
  assert(std::isnan(RelativeResidual(1,0)));

  const char* path="/tmp/ecal_summary_moments_test.root";
  {
    TFile f(path,"RECREATE");
    WriteHist(f,&full); WriteResidualStatistics(f,{&full,&empty});
  }
  {
    TFile f(path,"READ");
    auto* h=dynamic_cast<TH1D*>(f.Get("full")); assert(h);
    Near(h->GetMean(),1.5); Near(h->GetStdDev(),std::sqrt(5002.75));
    auto* s=dynamic_cast<TTree*>(f.Get("residual_statistics")); assert(s && s->GetEntries()==2);
    Long64_t count=0,invalid=0; double rmse=0;
    s->SetBranchAddress("count",&count); s->SetBranchAddress("invalid",&invalid); s->SetBranchAddress("rmse",&rmse);
    s->GetEntry(0); assert(count==4 && invalid==2); Near(rmse,std::sqrt(5005.));
  }
  std::remove(path);

  TTree t("events","Step2 fixture"); t.SetDirectory(nullptr);
  std::vector<float> truth={1,1,0}, pred={3,5,4};
  t.Branch("truth_eincident",&truth); t.Branch("pred_oracle_eincident",&pred); t.Fill();
  Step2(&t,"/tmp","ecal_summary_test");
  {
    TFile f("/tmp/ecal_summary_test_step2_diagnostics.root","READ");
    auto* s=dynamic_cast<TTree*>(f.Get("residual_statistics")); assert(s && s->GetEntries()==2);
    double mean=0,stddev=0,rmse=0; Long64_t count=0,invalid=0,overflow=0;
    s->SetBranchAddress("mean",&mean); s->SetBranchAddress("stddev",&stddev); s->SetBranchAddress("rmse",&rmse);
    s->SetBranchAddress("count",&count); s->SetBranchAddress("invalid",&invalid); s->SetBranchAddress("overflow",&overflow);
    s->GetEntry(1); Near(mean,3); Near(stddev,1); Near(rmse,std::sqrt(10.));
    assert(count==2 && invalid==1 && overflow==2);
  }
  std::remove("/tmp/ecal_summary_test_step2_diagnostics.root");
  std::cout << "Summary statistics tests passed.\n";
}
