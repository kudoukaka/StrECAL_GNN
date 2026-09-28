// reconstruct_ecal_clusters_from_eval_root.cxx
//
// Step-1/Step-2 validation for the multi-object hierarchical ECAL GNN.
// Adds two focused Step-1 diagnostics:
//   (1) truth-object split by whether the same primary has a delayed window;
//   (2) prompt-window energy/activity-weighted centroid position baseline.
//
// The position baseline requires these eval-ROOT branches:
//   window_centroid_y_mm, window_centroid_z_mm
// produced by the companion evaluate_ecal_gnn.py update.
//
// Usage:
//   root -l -q 'reconstruct_ecal_clusters_from_eval_root.cxx("eval_test_epoch_XXX.root")'

#include <TFile.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TSystem.h>
#include <array>
#include <map>
#include <stdexcept>
#include <TH1D.h>
#include <TH2D.h>
#include <TStyle.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

bool Has(TTree* t, const char* n) { return t && t->GetBranch(n); }

std::string Dir(const std::string& p) {
  const auto q = p.find_last_of('/');
  return q == std::string::npos ? "." : p.substr(0, q);
}

std::string Stem(const std::string& p) {
  const auto q = p.find_last_of('/');
  std::string s = q == std::string::npos ? p : p.substr(q + 1);
  if (s.size() > 5 && s.substr(s.size() - 5) == ".root") s.resize(s.size() - 5);
  return s;
}

bool Finite(double x) { return std::isfinite(x); }

double AngleDeg(double ax, double ay, double az,
                double bx, double by, double bz) {
  if (!Finite(ax)||!Finite(ay)||!Finite(az)||!Finite(bx)||!Finite(by)||!Finite(bz))
    return std::numeric_limits<double>::quiet_NaN();
  const double an = std::sqrt(ax*ax + ay*ay + az*az);
  const double bn = std::sqrt(bx*bx + by*by + bz*bz);
  if (an <= 0.0 || bn <= 0.0) return std::numeric_limits<double>::quiet_NaN();
  double c = (ax*bx + ay*by + az*bz) / (an*bn);
  c = std::max(-1.0, std::min(1.0, c));
  return std::acos(c) * 180.0 / M_PI;
}

double ThetaDeg(double x, double y, double z) {
  if (!Finite(x)||!Finite(y)||!Finite(z)) return std::numeric_limits<double>::quiet_NaN();
  const double r = std::sqrt(x*x + y*y + z*z);
  if (r <= 0.0) return std::numeric_limits<double>::quiet_NaN();
  double c = x / r; // polar axis is +x for the COMET ECAL geometry
  c = std::max(-1.0, std::min(1.0, c));
  return std::acos(c) * 180.0 / M_PI;
}

double PhiDeg(double /*x*/, double y, double z) {
  return std::atan2(z, y) * 180.0 / M_PI;
}

double WrapDeg(double x) {
  if (!Finite(x)) return std::numeric_limits<double>::quiet_NaN();
  while (x > 180.0) x -= 360.0;
  while (x <= -180.0) x += 360.0;
  return x;
}

double RelativeResidual(double prediction, double truth) {
  return Finite(prediction) && Finite(truth) && std::abs(truth)>1e-9
      ? (prediction-truth)/truth : std::numeric_limits<double>::quiet_NaN();
}

void PrintMetric(const std::string& name, double value, const std::string& unit="") {
  std::cout << "  " << std::left << std::setw(33) << name
            << " : " << std::right << std::setw(12) << std::fixed << std::setprecision(4)
            << value;
  if (!unit.empty()) std::cout << " " << unit;
  std::cout << "\n";
}

// Fill context is the truth object's species, or the source window's primary.
// Unknown/unassigned primaries contribute only to total (never guessed from reco).
int primaryCategory = 0;
const char* categoryNames[] = {"total", "electron", "muon", "pion"};
int Category(Long64_t pid) {
  switch (std::abs(pid)) { case 11: return 1; case 13: return 2;
    case 211: case 111: return 3; default: return 0; }
}
struct SplitHistograms {
  std::array<std::unique_ptr<TH1>,3> species;
  virtual ~SplitHistograms() = default;
  void Init(const TH1& h) {
    for (int i=1;i<4;++i) {
      species[i-1].reset(dynamic_cast<TH1*>(h.Clone((std::string(h.GetName())+"_"+categoryNames[i]).c_str())));
      species[i-1]->SetDirectory(nullptr);
      species[i-1]->SetTitle((std::string(h.GetTitle())+" ["+categoryNames[i]+"]").c_str());
    }
  }
};
struct Split1D : TH1D, SplitHistograms {
  Split1D(const char* n,const char* t,int b,double lo,double hi): TH1D(n,t,b,lo,hi) {
    SetDirectory(nullptr); SetStatOverflows(TH1::kConsider); Init(*this);
  }
  Int_t Fill(Double_t x) override { return Fill(x,1.); }
  Int_t Fill(Double_t x,Double_t w) override {
    if (primaryCategory>0) species[primaryCategory-1]->Fill(x,w);
    return TH1D::Fill(x,w);
  }
};
struct Split2D : TH2D, SplitHistograms {
  Split2D(const char* n,const char* t,int bx,double lx,double hx,int by,double ly,double hy)
      : TH2D(n,t,bx,lx,hx,by,ly,hy) { SetDirectory(nullptr); Init(*this); }
  Int_t Fill(Double_t x,Double_t y) override { return Fill(x,y,1.); }
  Int_t Fill(Double_t x,Double_t y,Double_t w) override {
    if (primaryCategory>0) static_cast<TH2*>(species[primaryCategory-1].get())->Fill(x,y,w);
    return TH2D::Fill(x,y,w);
  }
};
std::string plotDirectory;

// Accumulate unbinned residuals independently of histogram display ranges.
// Welford's update keeps the variance stable even for a large nonzero bias.
struct ResidualHistogram : Split1D {
  Long64_t count = 0, invalid = 0, underflow = 0, overflow = 0;
  long double mean = 0.0L, m2 = 0.0L;

  ResidualHistogram(const char* name, const char* title, int bins, double low, double high)
      : Split1D(name, title, bins, low, high) {
    SetDirectory(nullptr);
    // Saved ROOT histograms also include tails in their fill-time statistics.
    SetStatOverflows(TH1::kConsider);
  }

  Int_t Fill(Double_t value) override {
    if (!Finite(value)) { ++invalid; return -1; }
    ++count;
    if (value < GetXaxis()->GetXmin()) ++underflow;
    if (value >= GetXaxis()->GetXmax()) ++overflow;
    const long double delta = value - mean;
    mean += delta / count;
    m2 += delta * (value - mean);
    return Split1D::Fill(value);
  }

  double Mean() const { return count ? double(mean) : std::numeric_limits<double>::quiet_NaN(); }
  double StdDev() const {
    return count ? double(std::sqrt(std::max(0.0L, m2 / count))) : std::numeric_limits<double>::quiet_NaN();
  }
  double RMSE() const { return std::hypot(Mean(), StdDev()); }
};

void WriteResidualStatistics(TFile& f, const std::vector<ResidualHistogram*>& histograms) {
  f.cd();
  TTree stats("residual_statistics", "Unbinned finite residuals; independent of display range");
  std::string name;
  Long64_t count, invalid, underflow, overflow;
  double mean, stddev, rmse, low, high;
  stats.Branch("histogram", &name);
  stats.Branch("count", &count); stats.Branch("invalid", &invalid);
  stats.Branch("underflow", &underflow); stats.Branch("overflow", &overflow);
  stats.Branch("mean", &mean); stats.Branch("stddev", &stddev); stats.Branch("rmse", &rmse);
  stats.Branch("display_low", &low); stats.Branch("display_high", &high);
  for (const auto* h : histograms) {
    name=h->GetName(); count=h->count; invalid=h->invalid;
    underflow=h->underflow; overflow=h->overflow;
    mean=h->Mean(); stddev=h->StdDev(); rmse=h->RMSE();
    low=h->GetXaxis()->GetXmin(); high=h->GetXaxis()->GetXmax();
    stats.Fill();
  }
  stats.Write();
}

void PrintResidualTails(const std::vector<ResidualHistogram*>& histograms) {
  std::cout << "\n---------------- RESIDUAL COVERAGE ------------------------\n"
            << "Finite tails are INCLUDED in bias/std/RMSE. Display bins use [low, high).\n"
            << "Only histograms with tails or invalid values are listed (undefined relative errors count as invalid).\n";
  bool any=false;
  for (const auto* h : histograms) {
    if (!h->underflow && !h->overflow && !h->invalid) continue;
    any=true;
    std::cout << "  " << h->GetName() << ": N=" << h->count
              << " range=[" << h->GetXaxis()->GetXmin() << "," << h->GetXaxis()->GetXmax()
              << ") under=" << h->underflow << " over=" << h->overflow
              << " outside=" << (h->count ? 100.0*(h->underflow+h->overflow)/h->count : 0.0)
              << "% invalid=" << h->invalid << "\n";
  }
  if (!any) std::cout << "  No out-of-range or invalid residuals.\n";
}

void WriteHist(TFile& f, TH1* h) {
  if (!h) return;
  f.cd();
  h->Write(); // Keep the original total name for existing consumers.
  auto* split = dynamic_cast<SplitHistograms*>(h);
  if (!split) return;
  TCanvas canvas((std::string(h->GetName())+"_by_primary").c_str(),h->GetTitle(),1400,1000);
  canvas.Divide(2,2);
  std::unique_ptr<TH1> total(dynamic_cast<TH1*>(h->Clone((std::string(h->GetName())+"_total").c_str())));
  total->SetDirectory(nullptr);
  total->SetTitle((std::string(h->GetTitle())+" [total]").c_str());
  for (int i=0;i<4;++i) {
    TH1* panel = i ? split->species[i-1].get() : total.get();
    panel->Write(); canvas.cd(i+1);
    panel->Draw(panel->GetDimension()==2 ? "COLZ" : "HIST");
  }
  canvas.Write();
  if (!plotDirectory.empty()) canvas.SaveAs((plotDirectory+"/"+h->GetName()+"_by_primary.png").c_str());
}

struct ResidualSummary {
  std::unique_ptr<ResidualHistogram> t0;
  std::unique_ptr<ResidualHistogram> y;
  std::unique_ptr<ResidualHistogram> z;
  std::unique_ptr<ResidualHistogram> yzRadial;

  ResidualSummary(const std::string& tag, const std::string& title)
      : t0(std::make_unique<ResidualHistogram>((tag+"_t0_residual_ns").c_str(),
                                  (title+" t0 residual;pred-truth [ns];Objects").c_str(),
                                  800, -200.0, 200.0)),
        y(std::make_unique<ResidualHistogram>((tag+"_y_residual_mm").c_str(),
                                 (title+" Y residual;pred-truth [mm];Objects").c_str(),
                                 600, -150.0, 150.0)),
        z(std::make_unique<ResidualHistogram>((tag+"_z_residual_mm").c_str(),
                                 (title+" Z residual;pred-truth [mm];Objects").c_str(),
                                 600, -150.0, 150.0)),
        yzRadial(std::make_unique<ResidualHistogram>((tag+"_yz_radial_error_mm").c_str(),
                                        (title+" YZ radial error;#sqrt{#Deltay^{2}+#Deltaz^{2}} [mm];Objects").c_str(),
                                        400, 0.0, 200.0)) {}

  void Fill(double dt, double dy, double dz) {
    t0->Fill(dt); y->Fill(dy); z->Fill(dz);
    yzRadial->Fill(std::hypot(dy,dz));
  }

  void Write(TFile& f) {
    WriteHist(f,t0.get()); WriteHist(f,y.get()); WriteHist(f,z.get()); WriteHist(f,yzRadial.get());
  }
};

struct PositionBaseline {
  std::unique_ptr<ResidualHistogram> yAll = std::make_unique<ResidualHistogram>(
      "baseline_prompt_centroid_y_residual_all_mm",
      "Prompt-window centroid baseline Y residual, all truth objects;centroid-truth [mm];Objects",600,-150,150);
  std::unique_ptr<ResidualHistogram> zAll = std::make_unique<ResidualHistogram>(
      "baseline_prompt_centroid_z_residual_all_mm",
      "Prompt-window centroid baseline Z residual, all truth objects;centroid-truth [mm];Objects",600,-150,150);
  std::unique_ptr<ResidualHistogram> yzAll = std::make_unique<ResidualHistogram>(
      "baseline_prompt_centroid_yz_error_all_mm",
      "Prompt-window centroid baseline YZ error, all truth objects;#Delta r_{YZ} [mm];Objects",400,0,200);

  std::unique_ptr<ResidualHistogram> yMatched = std::make_unique<ResidualHistogram>(
      "baseline_prompt_centroid_y_residual_matched_mm",
      "Prompt-window centroid baseline Y residual, GNN-matched sample;centroid-truth [mm];Objects",600,-150,150);
  std::unique_ptr<ResidualHistogram> zMatched = std::make_unique<ResidualHistogram>(
      "baseline_prompt_centroid_z_residual_matched_mm",
      "Prompt-window centroid baseline Z residual, GNN-matched sample;centroid-truth [mm];Objects",600,-150,150);
  std::unique_ptr<ResidualHistogram> yzMatched = std::make_unique<ResidualHistogram>(
      "baseline_prompt_centroid_yz_error_matched_mm",
      "Prompt-window centroid baseline YZ error, GNN-matched sample;#Delta r_{YZ} [mm];Objects",400,0,200);

  long long validAll = 0;
  long long validMatched = 0;

  void Fill(double cy, double cz, double ty, double tz, bool matched) {
    if (!Finite(cy) || !Finite(cz) || !Finite(ty) || !Finite(tz)) {
      const double nan=std::numeric_limits<double>::quiet_NaN();
      yAll->Fill(nan); zAll->Fill(nan); yzAll->Fill(nan);
      if (matched) { yMatched->Fill(nan); zMatched->Fill(nan); yzMatched->Fill(nan); }
      return;
    }
    const double dy = cy - ty;
    const double dz = cz - tz;
    yAll->Fill(dy); zAll->Fill(dz); yzAll->Fill(std::sqrt(dy*dy+dz*dz));
    ++validAll;
    if (matched) {
      yMatched->Fill(dy); zMatched->Fill(dz); yzMatched->Fill(std::sqrt(dy*dy+dz*dz));
      ++validMatched;
    }
  }

  void Write(TFile& f) {
    for (TH1* h : {yAll.get(),zAll.get(),yzAll.get(),yMatched.get(),zMatched.get(),yzMatched.get()}) WriteHist(f,h);
  }
};

bool TruthObjectHasDelayedWindow(
    int promptWindow,
    const TTreeReaderArray<Long64_t>& assocSrc,
    const TTreeReaderArray<Long64_t>& assocDst,
    const TTreeReaderArray<Long64_t>& assocTarget,
    const TTreeReaderArray<Long64_t>& windowRoleTarget) {

  if (promptWindow < 0 || promptWindow >= static_cast<int>(windowRoleTarget.GetSize())) return false;
  const size_t n = std::min({assocSrc.GetSize(), assocDst.GetSize(), assocTarget.GetSize()});
  for (size_t k=0; k<n; ++k) {
    if (assocTarget[k] == 0) continue; // truth: not the same primary
    const int s = static_cast<int>(assocSrc[k]);
    const int d = static_cast<int>(assocDst[k]);
    int other = -1;
    if (s == promptWindow) other = d;
    else if (d == promptWindow) other = s;
    else continue;
    if (other >= 0 && other < static_cast<int>(windowRoleTarget.GetSize()) && windowRoleTarget[other] == 2)
      return true;
  }
  return false;
}

struct PrimaryReader {
  std::unique_ptr<TTreeReaderArray<Long64_t>> pid, primary, windows, edgeSrc;
  PrimaryReader(TTree* t,TTreeReader& r) {
    if (Has(t,"truth_pid")) pid=std::make_unique<TTreeReaderArray<Long64_t>>(r,"truth_pid");
    else std::cerr << "[warning] truth_pid absent: species panels are empty.\n";
    if (Has(t,"truth_primary_id")) primary=std::make_unique<TTreeReaderArray<Long64_t>>(r,"truth_primary_id");
    if (Has(t,"window_primary_id")) windows=std::make_unique<TTreeReaderArray<Long64_t>>(r,"window_primary_id");
    if (!windows && Has(t,"window_role_target"))
      std::cerr << "[warning] window_primary_id absent: window/pair/edge species panels require a new evaluation ROOT.\n";
    if (Has(t,"event_edge_src")) edgeSrc=std::make_unique<TTreeReaderArray<Long64_t>>(r,"event_edge_src");
  }
  int Object(size_t j) const { return pid && j<pid->GetSize() ? Category((*pid)[j]) : 0; }
  int Window(Long64_t w) const {
    if (!windows || !primary || w<0 || size_t(w)>=windows->GetSize() || (*windows)[w]<0) return 0;
    for(size_t j=0;j<primary->GetSize();++j) if((*primary)[j]==(*windows)[w]) return Object(j);
    return 0;
  }
};

void Step1(TTree* t, const std::string& outDir, const std::string& prefix) {
  const char* required[] = {
    "n_truth_object","n_reco_object","n_matched","n_energy_cut_rejected",
    "truth_t0","truth_y","truth_z","truth_dx","truth_dy","truth_dz","truth_ecluster","truth_emax",
    "pred_oracle_ecluster",
    "reco_t0","reco_y","reco_z","reco_dx","reco_dy","reco_dz","reco_ecluster","reco_emax",
    "reco_match_truth_index","truth_match_reco_index","truth_prompt_window",
    "association_src","association_dst","association_target","association_prob",
    "window_role_target","window_role_pred","window_prompt_prob","event_edge_dt"
  };
  for (const char* n : required) {
    if (!Has(t,n)) { std::cerr << "[error] missing branch: " << n << "\n"; return; }
  }

  const bool haveBaseline = Has(t,"window_centroid_y_mm") && Has(t,"window_centroid_z_mm");
  if (!haveBaseline) {
    std::cerr << "[warning] eval ROOT has no window_centroid_y_mm/window_centroid_z_mm.\n"
              << "          Delayed-window split will run, but position baseline will be skipped.\n"
              << "          Re-run evaluate_ecal_gnn.py with the diagnostic centroid branches enabled.\n";
  }

  TTreeReader r(t);
  PrimaryReader categories(t,r);
  TTreeReaderValue<Int_t> nTruth(r,"n_truth_object"), nReco(r,"n_reco_object"),
                              nMatched(r,"n_matched"), nECut(r,"n_energy_cut_rejected");

  TTreeReaderArray<Float_t> truthT(r,"truth_t0"), truthY(r,"truth_y"), truthZ(r,"truth_z"),
                            truthDx(r,"truth_dx"), truthDy(r,"truth_dy"), truthDz(r,"truth_dz"),
                            truthE(r,"truth_ecluster"), truthEmax(r,"truth_emax"),
                            oracleE(r,"pred_oracle_ecluster"),
                            recoT(r,"reco_t0"), recoY(r,"reco_y"), recoZ(r,"reco_z"),
                            recoDx(r,"reco_dx"), recoDy(r,"reco_dy"), recoDz(r,"reco_dz"),
                            recoE(r,"reco_ecluster"), recoEmax(r,"reco_emax"),
                            assocProb(r,"association_prob"), windowPromptProb(r,"window_prompt_prob"),
                            eventEdgeDt(r,"event_edge_dt");

  TTreeReaderArray<Long64_t> recoMatch(r,"reco_match_truth_index"), truthMatch(r,"truth_match_reco_index"),
                               truthPromptWindow(r,"truth_prompt_window"),
                               assocSrc(r,"association_src"), assocDst(r,"association_dst"), assocTarget(r,"association_target"),
                               roleTarget(r,"window_role_target"), rolePred(r,"window_role_pred");

  std::unique_ptr<TTreeReaderArray<Float_t>> winCentroidY;
  std::unique_ptr<TTreeReaderArray<Float_t>> winCentroidZ;
  if (haveBaseline) {
    winCentroidY = std::make_unique<TTreeReaderArray<Float_t>>(r,"window_centroid_y_mm");
    winCentroidZ = std::make_unique<TTreeReaderArray<Float_t>>(r,"window_centroid_z_mm");
  }

  auto hT = std::make_unique<ResidualHistogram>("matched_t0_residual_ns","Matched t0 residual;pred-truth [ns];Objects",1000,-50,50);
  auto hY = std::make_unique<ResidualHistogram>("matched_y_residual_mm","Matched Y residual;pred-truth [mm];Objects",400,-100,100);
  auto hZ = std::make_unique<ResidualHistogram>("matched_z_residual_mm","Matched Z residual;pred-truth [mm];Objects",400,-100,100);
  auto hYZ = std::make_unique<Split2D>("matched_yz_radial_error_mm","Matched YZ residual map;#Delta y [mm];#Delta z [mm]",400,-100,100,400,-100,100);
  auto hRadial = std::make_unique<ResidualHistogram>("matched_yz_radial_distance_mm","Matched YZ radial distance;sqrt(dy^2+dz^2) [mm];Objects",400,0,200);
  auto hOpen = std::make_unique<ResidualHistogram>("matched_opening_angle_deg","Matched opening-angle error;angle [deg];Objects",360,0,90);
  auto hTheta = std::make_unique<ResidualHistogram>("matched_theta_residual_deg","Matched #theta residual;pred-truth [deg];Objects",360,-90,90);
  auto hPhi = std::make_unique<ResidualHistogram>("matched_phi_residual_deg","Matched #phi residual;pred-truth [deg];Objects",720,-180,180);
  auto hAngle = std::make_unique<Split2D>("matched_angle_residual_deg","Matched angle residual;#Delta #theta [deg];#Delta #phi",360,-90,90,720,-180,180);
  auto hEabs = std::make_unique<ResidualHistogram>("matched_ecluster_residual_mev","Matched Ecluster residual;pred-truth [MeV];Objects",600,-100,100);
  auto hErel = std::make_unique<ResidualHistogram>("matched_ecluster_relative_residual","Matched Ecluster relative residual;(pred-truth)/truth;Objects",600,-2,2);
  auto hEmaxAbs = std::make_unique<ResidualHistogram>("matched_emax_residual_mev","Matched Emax residual;pred-truth [MeV];Objects",600,-100,100);
  auto hEmaxRel = std::make_unique<ResidualHistogram>("matched_emax_relative_residual","Matched Emax relative residual;(pred-truth)/truth;Objects",600,-2,2);
  auto hOracleErel = std::make_unique<ResidualHistogram>("oracle_ecluster_relative_residual","Oracle Ecluster relative residual;(pred-truth)/truth;Truth objects",600,-2,2);

  ResidualSummary noDelayed("no_delayed","Matched objects WITHOUT delayed same-primary window");
  ResidualSummary withDelayed("with_delayed","Matched objects WITH delayed same-primary window");
  PositionBaseline baseline;

  auto hAssocSame = std::make_unique<Split1D>("association_prob_truth_same","Association probability, truth same primary;P(same);Pairs",100,0,1);
  auto hAssocDiff = std::make_unique<Split1D>("association_prob_truth_different","Association probability, truth different primary;P(same);Pairs",100,0,1);
  auto hRoleConf = std::make_unique<Split2D>("window_role_confusion","Window-role confusion;truth;prediction",3,-0.5,2.5,3,-0.5,2.5);
  auto hAssocConf = std::make_unique<Split2D>("association_confusion","Association confusion @0.5;truth;prediction",2,-0.5,1.5,2,-0.5,1.5);
  auto hEdgeDt = std::make_unique<Split1D>("event_edge_dt","Directed event-edge #Deltat;#Deltat;Edges",500,-50,5000);

  long long totalTruth=0,totalReco=0,totalMatched=0,totalECut=0;
  long long truthNoDelay=0,truthWithDelay=0,matchNoDelay=0,matchWithDelay=0;
  long long assocTP=0,assocFP=0,assocTN=0,assocFN=0;
  long long roleCorrect=0,roleTotal=0,promptTP=0,promptFN=0,promptFP=0;
  long long directedEdges=0,negativeEdges=0,hybridViolations=0;

  while (r.Next()) {
    totalTruth += *nTruth; totalReco += *nReco; totalMatched += *nMatched; totalECut += *nECut;

    const size_t npair = std::min({assocTarget.GetSize(),assocProb.GetSize(),assocSrc.GetSize(),assocDst.GetSize()});
    for (size_t k=0;k<npair;++k) {
      primaryCategory = categories.Window(assocSrc[k]);
      const bool truthSame = assocTarget[k] != 0;
      const bool predSame = assocProb[k] >= 0.5;
      hAssocConf->Fill(truthSame?1:0,predSame?1:0);
      if (truthSame) hAssocSame->Fill(assocProb[k]); else hAssocDiff->Fill(assocProb[k]);
      if (truthSame && predSame) ++assocTP;
      else if (!truthSame && predSame) ++assocFP;
      else if (!truthSame && !predSame) ++assocTN;
      else ++assocFN;
    }

    const size_t nw = std::min({roleTarget.GetSize(),rolePred.GetSize(),windowPromptProb.GetSize()});
    for (size_t w=0;w<nw;++w) {
      primaryCategory = categories.Window(w);
      const int tr = static_cast<int>(roleTarget[w]);
      const int pr = static_cast<int>(rolePred[w]);
      if (tr>=0 && tr<3 && pr>=0 && pr<3) hRoleConf->Fill(tr,pr);
      if (tr==pr) ++roleCorrect;
      ++roleTotal;
      if (tr==1 && pr==1) ++promptTP;
      else if (tr==1 && pr!=1) ++promptFN;
      else if (tr!=1 && pr==1) ++promptFP;
    }

    for (size_t k=0;k<eventEdgeDt.GetSize();++k) {
      primaryCategory = categories.edgeSrc && k<categories.edgeSrc->GetSize() ? categories.Window((*categories.edgeSrc)[k]) : 0;
      const double dt=eventEdgeDt[k];
      if (!Finite(dt)) continue;
      hEdgeDt->Fill(dt); ++directedEdges;
      if (dt < -1.0e-6) {
        ++negativeEdges;
        if (dt < -30.0 - 1.0e-6) ++hybridViolations;
      }
    }

    // Per-truth-object delayed classification and position baseline.
    const size_t nobj = std::min({truthT.GetSize(),truthY.GetSize(),truthZ.GetSize(),truthPromptWindow.GetSize()});
    std::vector<unsigned char> hasDelayed(nobj,0);
    for (size_t j=0;j<nobj;++j) {
      primaryCategory = categories.Object(j);
      const int pw = static_cast<int>(truthPromptWindow[j]);
      hasDelayed[j] = TruthObjectHasDelayedWindow(pw,assocSrc,assocDst,assocTarget,roleTarget) ? 1 : 0;
      if (hasDelayed[j]) ++truthWithDelay; else ++truthNoDelay;
      const bool isMatched = (j < truthMatch.GetSize() && truthMatch[j] >= 0);
      if (haveBaseline && pw >= 0 &&
          pw < static_cast<int>(winCentroidY->GetSize()) &&
          pw < static_cast<int>(winCentroidZ->GetSize())) {
        baseline.Fill((*winCentroidY)[pw],(*winCentroidZ)[pw],truthY[j],truthZ[j],isMatched);
      }
    }

    for (size_t j=0;j<std::min(truthE.GetSize(),oracleE.GetSize());++j) {
      primaryCategory = categories.Object(j);
      hOracleErel->Fill(RelativeResidual(oracleE[j],truthE[j]));
    }

    const size_t nr = std::min({recoMatch.GetSize(),recoT.GetSize(),recoY.GetSize(),recoZ.GetSize(),
                                recoDx.GetSize(),recoDy.GetSize(),recoDz.GetSize(),recoE.GetSize(),recoEmax.GetSize()});
    for (size_t i=0;i<nr;++i) {
      const int j = static_cast<int>(recoMatch[i]);
      if (j<0 || j>=static_cast<int>(nobj) ||
          j>=static_cast<int>(truthDx.GetSize()) || j>=static_cast<int>(truthDy.GetSize()) || j>=static_cast<int>(truthDz.GetSize()) ||
          j>=static_cast<int>(truthE.GetSize()) || j>=static_cast<int>(truthEmax.GetSize())) continue;

      primaryCategory = categories.Object(j);
      const double dt = recoT[i]-truthT[j];
      const double dy = recoY[i]-truthY[j];
      const double dz = recoZ[i]-truthZ[j];
      hT->Fill(dt); hY->Fill(dy); hZ->Fill(dz);
      hRadial->Fill(std::hypot(dy,dz));
      if (Finite(dy) && Finite(dz)) hYZ->Fill(dy, dz);

      const double open = AngleDeg(recoDx[i],recoDy[i],recoDz[i],truthDx[j],truthDy[j],truthDz[j]);
      hOpen->Fill(open);
      const double thP=ThetaDeg(recoDx[i],recoDy[i],recoDz[i]);
      const double thT=ThetaDeg(truthDx[j],truthDy[j],truthDz[j]);
      hTheta->Fill(thP-thT);
      const double phP=PhiDeg(recoDx[i],recoDy[i],recoDz[i]);
      const double phT=PhiDeg(truthDx[j],truthDy[j],truthDz[j]);
      hPhi->Fill(WrapDeg(phP-phT));
      if (Finite(thP)&&Finite(thT)&&Finite(phP)&&Finite(phT)) hAngle->Fill(thP-thT,WrapDeg(phP-phT));

      hEabs->Fill(recoE[i]-truthE[j]);
      hErel->Fill(RelativeResidual(recoE[i],truthE[j]));
      hEmaxAbs->Fill(recoEmax[i]-truthEmax[j]);
      hEmaxRel->Fill(RelativeResidual(recoEmax[i],truthEmax[j]));

      if (hasDelayed[j]) {
        ++matchWithDelay;
        withDelayed.Fill(dt,dy,dz);
      } else {
        ++matchNoDelay;
        noDelayed.Fill(dt,dy,dz);
      }
    }
  }

  const std::string out = outDir + "/" + prefix + "_step1_diagnostics.root";
  const std::vector<ResidualHistogram*> residuals = {
    hT.get(),hY.get(),hZ.get(),hRadial.get(),hOpen.get(),hTheta.get(),hPhi.get(),
    hEabs.get(),hErel.get(),hEmaxAbs.get(),hEmaxRel.get(),hOracleErel.get(),
    noDelayed.t0.get(),noDelayed.y.get(),noDelayed.z.get(),noDelayed.yzRadial.get(),
    withDelayed.t0.get(),withDelayed.y.get(),withDelayed.z.get(),withDelayed.yzRadial.get(),
    baseline.yAll.get(),baseline.zAll.get(),baseline.yzAll.get(),
    baseline.yMatched.get(),baseline.zMatched.get(),baseline.yzMatched.get()
  };
  TFile fout(out.c_str(),"RECREATE");
  for (TH1* h : {static_cast<TH1*>(hT.get()),
                 static_cast<TH1*>(hY.get()),
                 static_cast<TH1*>(hZ.get()),
                 static_cast<TH1*>(hYZ.get()),
                 static_cast<TH1*>(hRadial.get()),
                 static_cast<TH1*>(hOpen.get()),
                 static_cast<TH1*>(hTheta.get()),
                 static_cast<TH1*>(hPhi.get()),
                 static_cast<TH1*>(hAngle.get()),
                 static_cast<TH1*>(hEabs.get()),
                 static_cast<TH1*>(hErel.get()),
                 static_cast<TH1*>(hEmaxAbs.get()),
                 static_cast<TH1*>(hEmaxRel.get()),
                 static_cast<TH1*>(hOracleErel.get()),
                 static_cast<TH1*>(hAssocSame.get()),
                 static_cast<TH1*>(hAssocDiff.get()),
                 static_cast<TH1*>(hRoleConf.get()),
                 static_cast<TH1*>(hAssocConf.get()),
                 static_cast<TH1*>(hEdgeDt.get())}) WriteHist(fout,h);
  noDelayed.Write(fout); withDelayed.Write(fout); baseline.Write(fout);
  WriteResidualStatistics(fout,residuals);
  fout.Close();

  const double yzCombined = std::sqrt(hY->StdDev()*hY->StdDev()+hZ->StdDev()*hZ->StdDev());

  std::cout << "\n========================================================\n"
            << "          ECAL STEP-1 RECONSTRUCTION SUMMARY\n"
            << "========================================================\n";
  std::cout << "Statistics: unbinned finite residuals, including display underflow/overflow; no fit.\n"
            << "Resolution = population standard deviation about the mean; RMSE = sqrt(mean(residual^2)).\n"
            << "Matched statistics retain evaluation selection/matching cuts; oracle uses all truth objects.\n\n";

  std::cout << "---------------- MATCHED-OBJECT RESOLUTION ----------------\n";
  PrintMetric("t0 bias",hT->Mean(),"ns");
  PrintMetric("t0 std (resolution)",hT->StdDev(),"ns");
  PrintMetric("t0 RMSE",hT->RMSE(),"ns");
  PrintMetric("Y bias",hY->Mean(),"mm");
  PrintMetric("Y std (resolution)",hY->StdDev(),"mm");
  PrintMetric("Y RMSE",hY->RMSE(),"mm");
  PrintMetric("Z bias",hZ->Mean(),"mm");
  PrintMetric("Z std (resolution)",hZ->StdDev(),"mm");
  PrintMetric("Z RMSE",hZ->RMSE(),"mm");
  PrintMetric("YZ combined resolution",yzCombined,"mm");
  PrintMetric("YZ radial RMSE",hRadial->RMSE(),"mm");
  PrintMetric("YZ radial error mean",hRadial->Mean(),"mm");
  PrintMetric("Opening-angle error mean",hOpen->Mean(),"deg");
  PrintMetric("Opening-angle std",hOpen->StdDev(),"deg");
  PrintMetric("Opening-angle RMSE",hOpen->RMSE(),"deg");
  PrintMetric("Theta residual bias",hTheta->Mean(),"deg");
  PrintMetric("Theta std (resolution)",hTheta->StdDev(),"deg");
  PrintMetric("Theta RMSE",hTheta->RMSE(),"deg");
  PrintMetric("Phi residual bias",hPhi->Mean(),"deg");
  PrintMetric("Phi std (resolution)",hPhi->StdDev(),"deg");
  PrintMetric("Phi RMSE",hPhi->RMSE(),"deg");
  PrintMetric("Ecluster bias",hEabs->Mean(),"MeV");
  PrintMetric("Ecluster std (resolution)",hEabs->StdDev(),"MeV");
  PrintMetric("Ecluster RMSE",hEabs->RMSE(),"MeV");
  PrintMetric("Ecluster relative bias",100.0*hErel->Mean(),"%");
  PrintMetric("Ecluster relative std",100.0*hErel->StdDev(),"%");
  PrintMetric("Ecluster relative RMSE",100.0*hErel->RMSE(),"%");
  PrintMetric("Emax relative bias",100.0*hEmaxRel->Mean(),"%");
  PrintMetric("Emax relative std",100.0*hEmaxRel->StdDev(),"%");
  PrintMetric("Emax relative RMSE",100.0*hEmaxRel->RMSE(),"%");
  PrintMetric("Oracle Ecluster relative std",100.0*hOracleErel->StdDev(),"%");
  PrintMetric("Oracle Ecluster relative bias",100.0*hOracleErel->Mean(),"%");
  PrintMetric("Oracle Ecluster relative RMSE",100.0*hOracleErel->RMSE(),"%");

  std::cout << "\n---------------- DELAYED-WINDOW SPLIT ---------------------\n";
  std::cout << "Definition: WITH delayed = prompt window has at least one truth same-primary\n"
            << "            associated window with window_role_target==2.\n";
  std::cout << "  " << std::left << std::setw(28) << "Category"
            << std::right << std::setw(12) << "Truth" << std::setw(12) << "Matched"
            << std::setw(15) << "Efficiency" << "\n";
  auto printGroup=[&](const char* name,long long truth,long long match,ResidualSummary& h){
    std::cout << "  " << std::left << std::setw(28) << name
              << std::right << std::setw(12) << truth << std::setw(12) << match
              << std::setw(14) << std::fixed << std::setprecision(3) << (truth?100.0*double(match)/truth:0.0) << "%\n";
    PrintMetric(std::string(name)+" t0 bias",h.t0->Mean(),"ns");
    PrintMetric(std::string(name)+" t0 resolution",h.t0->StdDev(),"ns");
    PrintMetric(std::string(name)+" t0 RMSE",h.t0->RMSE(),"ns");
    PrintMetric(std::string(name)+" YZ combined res.",std::sqrt(h.y->StdDev()*h.y->StdDev()+h.z->StdDev()*h.z->StdDev()),"mm");
    PrintMetric(std::string(name)+" YZ radial mean",h.yzRadial->Mean(),"mm");
    PrintMetric(std::string(name)+" YZ radial RMSE",h.yzRadial->RMSE(),"mm");
  };
  printGroup("NO delayed",truthNoDelay,matchNoDelay,noDelayed);
  printGroup("WITH delayed",truthWithDelay,matchWithDelay,withDelayed);
  PrintMetric("Truth delayed-object fraction", totalTruth?100.0*double(truthWithDelay)/totalTruth:0.0, "%");

  std::cout << "\n---------------- POSITION BASELINE ------------------------\n";
  if (haveBaseline && baseline.validAll>0) {
    std::cout << "Baseline = energy/activity-weighted centroid of the truth prompt seed window.\n";
    PrintMetric("Baseline valid truth objects",baseline.validAll,"");
    PrintMetric("Baseline Y bias (all)",baseline.yAll->Mean(),"mm");
    PrintMetric("Baseline Y resolution (all)",baseline.yAll->StdDev(),"mm");
    PrintMetric("Baseline Z bias (all)",baseline.zAll->Mean(),"mm");
    PrintMetric("Baseline Z resolution (all)",baseline.zAll->StdDev(),"mm");
    PrintMetric("Baseline YZ combined (all)",std::sqrt(baseline.yAll->StdDev()*baseline.yAll->StdDev()+baseline.zAll->StdDev()*baseline.zAll->StdDev()),"mm");
    PrintMetric("Baseline YZ radial mean (all)",baseline.yzAll->Mean(),"mm");
    PrintMetric("Baseline YZ RMSE (all)",baseline.yzAll->RMSE(),"mm");
    std::cout << "\n  Same sample as GNN matched objects:\n";
    PrintMetric("Baseline valid matched objects",baseline.validMatched,"");
    const double baseMatchedCombined=std::sqrt(baseline.yMatched->StdDev()*baseline.yMatched->StdDev()+baseline.zMatched->StdDev()*baseline.zMatched->StdDev());
    PrintMetric("Baseline Y bias (matched)",baseline.yMatched->Mean(),"mm");
    PrintMetric("Baseline Y resolution (matched)",baseline.yMatched->StdDev(),"mm");
    PrintMetric("Baseline Z bias (matched)",baseline.zMatched->Mean(),"mm");
    PrintMetric("Baseline Z resolution (matched)",baseline.zMatched->StdDev(),"mm");
    PrintMetric("Baseline YZ combined (matched)",baseMatchedCombined,"mm");
    PrintMetric("Baseline YZ radial mean (matched)",baseline.yzMatched->Mean(),"mm");
    PrintMetric("Baseline YZ RMSE (matched)",baseline.yzMatched->RMSE(),"mm");
    if (yzCombined>0.0) PrintMetric("Baseline/GNN YZ res. ratio",baseMatchedCombined/yzCombined,"");
  } else {
    std::cout << "  [skipped] window centroid branches are absent or empty.\n";
  }

  std::cout << "\n---------------- OBJECT RECONSTRUCTION -------------------\n";
  std::cout << "  Truth objects                 : " << totalTruth << "\n"
            << "  Reco objects                  : " << totalReco << "\n"
            << "  Matched objects               : " << totalMatched << "\n"
            << "  Missed truth objects          : " << (totalTruth-totalMatched) << "\n"
            << "  Unmatched reco objects        : " << (totalReco-totalMatched) << "\n";
  PrintMetric("Object efficiency",totalTruth?100.0*double(totalMatched)/totalTruth:0.0,"%");
  PrintMetric("Object purity",totalReco?100.0*double(totalMatched)/totalReco:0.0,"%");
  std::cout << "  Energy-cut rejected objects   : " << totalECut << "\n";

  std::cout << "\n---------------- WINDOW ASSOCIATION ----------------------\n";
  const long long assocTot=assocTP+assocTN+assocFP+assocFN;
  PrintMetric("Association accuracy @0.5",assocTot?100.0*double(assocTP+assocTN)/assocTot:0.0,"%");
  PrintMetric("Same-primary efficiency",(assocTP+assocFN)?100.0*double(assocTP)/(assocTP+assocFN):0.0,"%");
  PrintMetric("Same-primary purity",(assocTP+assocFP)?100.0*double(assocTP)/(assocTP+assocFP):0.0,"%");
  PrintMetric("Different-primary rejection",(assocTN+assocFP)?100.0*double(assocTN)/(assocTN+assocFP):0.0,"%");
  PrintMetric("Window-role accuracy",roleTotal?100.0*double(roleCorrect)/roleTotal:0.0,"%");
  PrintMetric("Prompt-role efficiency",(promptTP+promptFN)?100.0*double(promptTP)/(promptTP+promptFN):0.0,"%");
  PrintMetric("Prompt-role purity",(promptTP+promptFP)?100.0*double(promptTP)/(promptTP+promptFP):0.0,"%");

  std::cout << "\n---------------- EVENT GRAPH CHECK -----------------------\n";
  std::cout << "  Directed event edges          : " << directedEdges << "\n"
            << "  Negative-dt edges             : " << negativeEdges << "  (allowed for |dt| <= 30 ns hybrid pairs)\n"
            << "  Hybrid direction violations   : " << hybridViolations << "  (negative dt < -30 ns; should be 0)\n";
  PrintMetric("Negative-dt fraction",directedEdges?100.0*double(negativeEdges)/directedEdges:0.0,"%");
  PrintResidualTails(residuals);
  std::cout << "\n[write] " << out << "\n";
}

void Step2(TTree* t, const std::string& outDir, const std::string& prefix) {
  if (!Has(t,"truth_eincident") || !Has(t,"pred_oracle_eincident")) {
    std::cerr << "[error] missing Step-2 incident-energy branches\n";
    return;
  }
  TTreeReader r(t);
  PrimaryReader categories(t,r);
  TTreeReaderArray<Float_t> truthE(r,"truth_eincident"), predE(r,"pred_oracle_eincident");
  auto hAbs=std::make_unique<ResidualHistogram>("incident_energy_residual_mev","Step-2 Eincident residual;pred-truth [MeV];Objects",600,-150,150);
  auto hRel=std::make_unique<ResidualHistogram>("incident_energy_relative_residual","Step-2 Eincident relative residual;(pred-truth)/truth;Objects",600,-2,2);
  while(r.Next()) {
    const size_t n=std::min(truthE.GetSize(),predE.GetSize());
    for(size_t i=0;i<n;++i) {
      primaryCategory = categories.Object(i);
      hAbs->Fill(predE[i]-truthE[i]);
      hRel->Fill(RelativeResidual(predE[i],truthE[i]));
    }
  }
  const std::string out=outDir+"/"+prefix+"_step2_diagnostics.root";
  TFile fout(out.c_str(),"RECREATE"); WriteHist(fout,hAbs.get()); WriteHist(fout,hRel.get());
  WriteResidualStatistics(fout,{hAbs.get(),hRel.get()}); fout.Close();
  std::cout << "[step2] All finite oracle residuals, including display underflow/overflow; no fit.\n";
  PrintMetric("Eincident bias",hAbs->Mean(),"MeV");
  PrintMetric("Eincident std (resolution)",hAbs->StdDev(),"MeV");
  PrintMetric("Eincident RMSE",hAbs->RMSE(),"MeV");
  PrintMetric("Eincident relative bias",100.0*hRel->Mean(),"%");
  PrintMetric("Eincident relative std",100.0*hRel->StdDev(),"%");
  PrintMetric("Eincident relative RMSE",100.0*hRel->RMSE(),"%");
  PrintResidualTails({hAbs.get(),hRel.get()});
  std::cout << "[write] " << out << "\n";
}

} // namespace

void reconstruct_ecal_clusters_from_eval_root_delayed_baseline(const char* inputFile) {
  gROOT->SetBatch(kTRUE);
  gStyle->SetOptStat(1110);
  std::unique_ptr<TFile> f(TFile::Open(inputFile,"READ"));
  if(!f || f->IsZombie()) { std::cerr << "[error] cannot open " << inputFile << "\n"; return; }
  TTree* t=dynamic_cast<TTree*>(f->Get("events"));
  if(!t || !Has(t,"reconstruction_step")) { std::cerr << "[error] missing events/reconstruction_step\n"; return; }
  Int_t step=0; t->SetBranchAddress("reconstruction_step",&step);
  if(t->GetEntries()<=0) { std::cerr << "[error] empty events tree\n"; return; }
  t->GetEntry(0); t->ResetBranchAddresses();
  const std::string in=inputFile, outDir=Dir(in), prefix=Stem(in);
  plotDirectory=outDir+"/"+prefix+"_figures";
  gSystem->mkdir(plotDirectory.c_str(),kTRUE);
  std::cout << "[input] " << in << "  step=" << step << "\n";
  if(step==1) Step1(t,outDir,prefix);
  else if(step==2) Step2(t,outDir,prefix);
  else std::cerr << "[error] unsupported reconstruction_step=" << step << "\n";
}

// Alias preserving the old invocation name if this file is renamed to
// reconstruct_ecal_clusters_from_eval_root.cxx.
void reconstruct_ecal_clusters_from_eval_root(const char* inputFile) {
  reconstruct_ecal_clusters_from_eval_root_delayed_baseline(inputFile);
}
