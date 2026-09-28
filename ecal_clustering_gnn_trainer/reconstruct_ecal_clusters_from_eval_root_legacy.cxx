// reconstruct_ecal_clusters_from_eval_root.cxx
//
// Read the per-seed-window ROOT output from evaluate_ecal_gnn.py, reconstruct
// ECAL clusters from predicted signal nodes and predicted same-cluster edges,
// then validate against truth clusters.
//
// Important in this version:
//   - Reco t0 / t0 position / energy use node observables written by
//     evaluate_ecal_gnn.py.  For truth-feature eval ROOT files, this macro
//     automatically prefers node_truth_time/node_truth_energy when present;
//     otherwise it falls back to node_t/node_energy for old reco-feature files.
//   - Truth t0 / t0 position / energy are looked up from full-event truth-cluster
//     branches truth_cluster_*; they are not recomputed from only the current
//     seed-window nodes.
//   - Splitting is still evaluated inside each given seed-window graph.
//
// Usage in ROOT:
//   root -l -q 'reconstruct_ecal_clusters_from_eval_root.cxx("/home/comet/davidz/ReconDev/ReconData/runs/ecal_clustering/<setup>/outputs/eval_test_epoch_036.root")'

#include <TCanvas.h>
#include <TFile.h>
#include <TH1.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <TEfficiency.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct DSU {
  std::vector<int> parent;
  std::vector<int> rank;

  explicit DSU(int n = 0) { reset(n); }

  void reset(int n) {
    parent.resize(n);
    rank.assign(n, 0);
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    if (parent[x] == x) return x;
    parent[x] = find(parent[x]);
    return parent[x];
  }

  void unite(int a, int b) {
    int ra = find(a);
    int rb = find(b);
    if (ra == rb) return;
    if (rank[ra] < rank[rb]) std::swap(ra, rb);
    parent[rb] = ra;
    if (rank[ra] == rank[rb]) ++rank[ra];
  }
};

template <typename T>
std::vector<T> CopyArray(TTreeReaderArray<T>& arr) {
  std::vector<T> out;
  out.reserve(arr.GetSize());
  for (const auto& v : arr) out.push_back(v);
  return out;
}

bool HasBranch(TTree* tree, const char* name) {
  return tree != nullptr && name != nullptr && tree->GetBranch(name) != nullptr;
}

template <typename T>
std::vector<T> CopyArrayIfPresent(const std::unique_ptr<TTreeReaderArray<T>>& arr) {
  if (!arr) return {};
  std::vector<T> out;
  out.reserve(arr->GetSize());
  for (const auto& v : *arr) out.push_back(v);
  return out;
}

bool HasNodeVectorSize(const std::vector<float>& v, int nNode) {
  return static_cast<int>(v.size()) == nNode;
}

std::vector<std::vector<int>> BuildPredictedClusters(
    int nNode,
    const std::vector<float>& nodeProb,
    const std::vector<unsigned char>& nodeValid,
    const std::vector<int>& edgeSrc,
    const std::vector<int>& edgeDst,
    const std::vector<float>& edgeProb,
    const std::vector<unsigned char>& edgeValid,
    double nodeThreshold,
    double edgeThreshold,
    int minRecoNodes) {
  std::vector<unsigned char> selected(nNode, 0);
  for (int i = 0; i < nNode; ++i) {
    selected[i] = (i < static_cast<int>(nodeValid.size()) && i < static_cast<int>(nodeProb.size()) &&
                   nodeValid[i] != 0 && nodeProb[i] > nodeThreshold) ? 1 : 0;
  }

  DSU dsu(nNode);
  const int nEdge = static_cast<int>(edgeSrc.size());
  for (int e = 0; e < nEdge; ++e) {
    const int s = edgeSrc[e];
    const int d = edgeDst[e];
    if (s < 0 || d < 0 || s >= nNode || d >= nNode) continue;
    if (!selected[s] || !selected[d]) continue;
    if (e >= static_cast<int>(edgeValid.size()) || e >= static_cast<int>(edgeProb.size())) continue;
    if (edgeValid[e] == 0 || edgeProb[e] <= edgeThreshold) continue;
    dsu.unite(s, d);
  }

  std::map<int, std::vector<int>> byRoot;
  for (int i = 0; i < nNode; ++i) {
    if (!selected[i]) continue;
    byRoot[dsu.find(i)].push_back(i);
  }

  std::vector<std::vector<int>> clusters;
  for (auto& kv : byRoot) {
    if (static_cast<int>(kv.second.size()) >= minRecoNodes) clusters.push_back(std::move(kv.second));
  }
  return clusters;
}

std::vector<std::vector<int>> BuildTruthClusters(
    int nNode,
    const std::vector<float>& nodeLabel,
    const std::vector<unsigned char>& nodeValid,
    const std::vector<long long>& nodeTruthCluster,
    const std::vector<int>& edgeSrc,
    const std::vector<int>& edgeDst,
    const std::vector<float>& edgeLabel,
    const std::vector<unsigned char>& edgeValid,
    std::vector<long long>& truthIdsOut) {
  truthIdsOut.clear();

  bool hasExplicitTruthId = false;
  for (int i = 0; i < nNode; ++i) {
    if (i < static_cast<int>(nodeValid.size()) && i < static_cast<int>(nodeLabel.size()) &&
        i < static_cast<int>(nodeTruthCluster.size()) &&
        nodeValid[i] != 0 && nodeLabel[i] > 0.5f && nodeTruthCluster[i] >= 0) {
      hasExplicitTruthId = true;
      break;
    }
  }

  if (hasExplicitTruthId) {
    std::map<long long, std::vector<int>> byTruthId;
    for (int i = 0; i < nNode; ++i) {
      if (i >= static_cast<int>(nodeValid.size()) || i >= static_cast<int>(nodeLabel.size()) ||
          i >= static_cast<int>(nodeTruthCluster.size())) continue;
      if (nodeValid[i] == 0 || nodeLabel[i] <= 0.5f) continue;
      const long long tid = nodeTruthCluster[i];
      if (tid < 0) continue;
      byTruthId[tid].push_back(i);
    }

    std::vector<std::vector<int>> clusters;
    for (auto& kv : byTruthId) {
      truthIdsOut.push_back(kv.first);
      clusters.push_back(std::move(kv.second));
    }
    return clusters;
  }

  // Fallback for older eval ROOT files without node_truth_cluster.
  DSU dsu(nNode);
  const int nEdge = static_cast<int>(edgeSrc.size());
  for (int e = 0; e < nEdge; ++e) {
    const int s = edgeSrc[e];
    const int d = edgeDst[e];
    if (s < 0 || d < 0 || s >= nNode || d >= nNode) continue;
    if (s >= static_cast<int>(nodeValid.size()) || d >= static_cast<int>(nodeValid.size())) continue;
    if (s >= static_cast<int>(nodeLabel.size()) || d >= static_cast<int>(nodeLabel.size())) continue;
    if (nodeValid[s] == 0 || nodeValid[d] == 0) continue;
    if (nodeLabel[s] <= 0.5f || nodeLabel[d] <= 0.5f) continue;
    if (e >= static_cast<int>(edgeValid.size()) || e >= static_cast<int>(edgeLabel.size())) continue;
    if (edgeValid[e] == 0 || edgeLabel[e] <= 0.5f) continue;
    dsu.unite(s, d);
  }

  std::map<int, std::vector<int>> byRoot;
  for (int i = 0; i < nNode; ++i) {
    if (i >= static_cast<int>(nodeValid.size()) || i >= static_cast<int>(nodeLabel.size())) continue;
    if (nodeValid[i] == 0 || nodeLabel[i] <= 0.5f) continue;
    byRoot[dsu.find(i)].push_back(i);
  }

  std::vector<std::vector<int>> clusters;
  long long fallbackId = 0;
  for (auto& kv : byRoot) {
    truthIdsOut.push_back(fallbackId++);
    clusters.push_back(std::move(kv.second));
  }
  return clusters;
}

int CountOverlap(const std::vector<int>& a, const std::vector<int>& b) {
  int n = 0;
  for (const int ia : a) {
    for (const int ib : b) {
      if (ia == ib) {
        ++n;
        break;
      }
    }
  }
  return n;
}

struct ClusterObservables {
  bool valid = false;
  int t0Node = -1;
  float t0 = std::numeric_limits<float>::quiet_NaN();
  float t0X = std::numeric_limits<float>::quiet_NaN();
  float t0Y = std::numeric_limits<float>::quiet_NaN();
  float t0Z = std::numeric_limits<float>::quiet_NaN();
  float energy = std::numeric_limits<float>::quiet_NaN();
};

ClusterObservables ComputeRecoObservables(
    const std::vector<int>& cluster,
    const std::vector<float>& nodeY,
    const std::vector<float>& nodeZ,
    const std::vector<float>& nodeT,
    const std::vector<float>& nodeEnergy) {
  ClusterObservables obs;
  if (cluster.empty()) return obs;

  obs.valid = true;
  obs.energy = 0.0f;
  float bestT = std::numeric_limits<float>::infinity();
  int bestNode = -1;

  for (const int iNode : cluster) {
    if (iNode < 0) continue;
    if (iNode < static_cast<int>(nodeEnergy.size()) && std::isfinite(nodeEnergy[iNode])) {
      obs.energy += nodeEnergy[iNode];
    }
    if (iNode < static_cast<int>(nodeT.size()) && nodeT[iNode] < bestT) {
      bestT = nodeT[iNode];
      bestNode = iNode;
    }
  }

  obs.t0Node = bestNode;
  if (bestNode >= 0) {
    obs.t0 = bestT;
    if (bestNode < static_cast<int>(nodeY.size())) obs.t0Y = nodeY[bestNode];
    if (bestNode < static_cast<int>(nodeZ.size())) obs.t0Z = nodeZ[bestNode];
  }
  return obs;
}

std::vector<ClusterObservables> ComputeRecoObservableList(
    const std::vector<std::vector<int>>& clusters,
    const std::vector<float>& nodeY,
    const std::vector<float>& nodeZ,
    const std::vector<float>& nodeT,
    const std::vector<float>& nodeEnergy) {
  std::vector<ClusterObservables> out;
  out.reserve(clusters.size());
  for (const auto& cluster : clusters) out.push_back(ComputeRecoObservables(cluster, nodeY, nodeZ, nodeT, nodeEnergy));
  return out;
}

std::unordered_map<long long, ClusterObservables> BuildTruthObservableMap(
    const std::vector<long long>& truthClusterId,
    const std::vector<float>& truthClusterEnergy,
    const std::vector<float>& truthClusterT0,
    const std::vector<float>& truthClusterT0X,
    const std::vector<float>& truthClusterT0Y,
    const std::vector<float>& truthClusterT0Z) {
  std::unordered_map<long long, ClusterObservables> out;
  const std::size_t n = truthClusterId.size();
  for (std::size_t i = 0; i < n; ++i) {
    ClusterObservables obs;
    obs.valid = true;
    if (i < truthClusterEnergy.size()) obs.energy = truthClusterEnergy[i];
    if (i < truthClusterT0.size()) obs.t0 = truthClusterT0[i];
    if (i < truthClusterT0X.size()) obs.t0X = truthClusterT0X[i];
    if (i < truthClusterT0Y.size()) obs.t0Y = truthClusterT0Y[i];
    if (i < truthClusterT0Z.size()) obs.t0Z = truthClusterT0Z[i];
    out[truthClusterId[i]] = obs;
  }
  return out;
}

std::unordered_map<long long, long long> BuildTruthSizeMap(
    const std::vector<long long>& truthClusterId,
    const std::vector<long long>& truthClusterSize) {
  std::unordered_map<long long, long long> out;
  const std::size_t n = std::min(truthClusterId.size(), truthClusterSize.size());
  for (std::size_t i = 0; i < n; ++i) out[truthClusterId[i]] = truthClusterSize[i];
  return out;
}

std::unordered_map<long long, long long> BuildTruthIntegerMap(
    const std::vector<long long>& truthClusterId,
    const std::vector<long long>& values) {
  std::unordered_map<long long, long long> out;
  const std::size_t n = std::min(truthClusterId.size(), values.size());
  for (std::size_t i = 0; i < n; ++i) out[truthClusterId[i]] = values[i];
  return out;
}

TString DirName(const char* path) {
  TString p(path ? path : "");
  const Ssiz_t slash = p.Last('/');
  if (slash < 0) return ".";
  return p(0, slash);
}

TString BaseStem(const char* path) {
  TString p(path ? path : "");
  const Ssiz_t slash = p.Last('/');
  if (slash >= 0) p = p(slash + 1, p.Length() - slash - 1);
  if (p.EndsWith(".root")) p.Remove(p.Length() - 5);
  return p;
}

TString ResolveOutputRoot(const char* inputRoot, const char* outputRoot) {
  TString out(outputRoot ? outputRoot : "");
  const bool autoName = out.IsNull() || out == "AUTO" || out == "auto" || out == "ecal_gnn_reco_clusters.root";
  if (!autoName) return out;
  return DirName(inputRoot) + "/ecal_gnn_reco_clusters_" + BaseStem(inputRoot) + ".root";
}

TString DefaultPlotDir(const char* outputRoot) {
  TString d(outputRoot);
  if (d.EndsWith(".root")) d.ReplaceAll(".root", "_plots");
  else d += "_plots";
  return d;
}

void SaveCanvas(TCanvas& c, const TString& plotDir, const char* name) {
  if (plotDir.IsNull()) return;
  c.SaveAs(plotDir + "/" + name + ".png");
  c.SaveAs(plotDir + "/" + name + ".pdf");
}

TH1D* MakeNormalizedClone(const TH1D& h, const char* cloneName) {
  TH1D* c = static_cast<TH1D*>(h.Clone(cloneName));
  c->SetDirectory(nullptr);
  if (c->Integral() > 0.0) c->Scale(1.0 / c->Integral());
  return c;
}

void DrawOverlayNormalized(
    const TH1D& hA,
    const char* labelA,
    const TH1D& hB,
    const char* labelB,
    const TString& plotDir,
    const char* canvasName,
    bool logY = false) {
  TString cloneNameA = TString(hA.GetName()) + "_norm_plot_a";
  TString cloneNameB = TString(hB.GetName()) + "_norm_plot_b";
  std::unique_ptr<TH1D> a(MakeNormalizedClone(hA, cloneNameA.Data()));
  std::unique_ptr<TH1D> b(MakeNormalizedClone(hB, cloneNameB.Data()));

  a->SetLineColor(kBlue + 1);
  a->SetLineWidth(2);
  b->SetLineColor(kRed + 1);
  b->SetLineWidth(2);
  const double maxY = std::max(a->GetMaximum(), b->GetMaximum());
  a->SetMaximum(logY ? std::max(1.0e-3, maxY * 5.0) : maxY * 1.25);
  if (logY) a->SetMinimum(1.0e-6);

  TCanvas c(canvasName, canvasName, 900, 700);
  c.SetGrid();
  if (logY) c.SetLogy();
  a->Draw("hist");
  b->Draw("hist same");
  TLegend leg(0.60, 0.75, 0.88, 0.88);
  leg.SetBorderSize(0);
  leg.SetFillStyle(0);
  leg.AddEntry(a.get(), labelA, "l");
  leg.AddEntry(b.get(), labelB, "l");
  leg.Draw();
  SaveCanvas(c, plotDir, canvasName);
}

void DrawSingle(TH1& h, const TString& plotDir, const char* canvasName, bool logY = false) {
  TCanvas c(canvasName, canvasName, 900, 700);
  c.SetGrid();
  if (logY) c.SetLogy();
  h.SetLineWidth(2);
  h.Draw("hist");
  SaveCanvas(c, plotDir, canvasName);
}

void DrawSingle2D(TH2& h, const TString& plotDir, const char* canvasName) {
  TCanvas c(canvasName, canvasName, 900, 750);
  c.SetGrid();
  c.SetRightMargin(0.15);
  h.Draw("colz");
  SaveCanvas(c, plotDir, canvasName);
}

}  // namespace

void reconstruct_ecal_clusters_from_eval_root(
    const char* inputRoot,
    const char* outputRoot = "ecal_gnn_reco_clusters.root",
    double nodeThreshold = 0.5,
    double edgeThreshold = 0.5,
    double minCompleteness = 0.5,
    double minPurity = 0.5,
    int minTruthNodes = 1,
    int minRecoNodes = 1,
    const char* plotDir = "",
    int minSplitOverlapNodes = 1,
    double minSplitCompleteness = 0.05) {
  TH1::AddDirectory(kFALSE);
  gStyle->SetOptStat(1110);

  std::unique_ptr<TFile> fin(TFile::Open(inputRoot, "READ"));
  if (!fin || fin->IsZombie()) {
    std::cerr << "[error] Cannot open input ROOT file: " << inputRoot << std::endl;
    return;
  }

  TTree* events = dynamic_cast<TTree*>(fin->Get("events"));
  if (!events) {
    std::cerr << "[error] Cannot find TTree 'events' in " << inputRoot << std::endl;
    return;
  }

  TTreeReader reader(events);
  TTreeReaderValue<long long> eventIndex(reader, "event_index");
  TTreeReaderValue<int> fileIndex(reader, "file_index");
  TTreeReaderValue<long long> graphIndexInFile(reader, "graph_index_in_file");
  TTreeReaderValue<int> nNodeBranch(reader, "n_node");

  const bool hasGraphSourceBranch = HasBranch(events, "graph_source");
  std::unique_ptr<TTreeReaderValue<int>> graphSourceBranch;
  if (hasGraphSourceBranch) {
    graphSourceBranch.reset(new TTreeReaderValue<int>(reader, "graph_source"));
  }
  const bool hasNodeTruthClusterBranch = HasBranch(events, "node_truth_cluster");
  const bool hasNodeTruthTimeBranch = HasBranch(events, "node_truth_time");
  const bool hasNodeTruthEnergyBranch = HasBranch(events, "node_truth_energy");
  const bool hasNodeRecoTBranch = HasBranch(events, "node_reco_t");
  const bool hasNodeRecoAmpBranch = HasBranch(events, "node_reco_amp");
  const bool hasNodeRecoAreaBranch = HasBranch(events, "node_reco_area");
  const bool hasTruthClusterIdBranch = HasBranch(events, "truth_cluster_id");
  const bool hasTruthClusterPrimaryIdBranch = HasBranch(events, "truth_cluster_primary_id");
  const bool hasTruthClusterTimeSliceBranch = HasBranch(events, "truth_cluster_time_slice");
  const bool hasTruthClusterSizeBranch = HasBranch(events, "truth_cluster_size");
  const bool hasTruthClusterEnergyBranch = HasBranch(events, "truth_cluster_energy");
  const bool hasTruthClusterT0Branch = HasBranch(events, "truth_cluster_t0");
  const bool hasTruthClusterT0XBranch = HasBranch(events, "truth_cluster_t0_x");
  const bool hasTruthClusterT0YBranch = HasBranch(events, "truth_cluster_t0_y");
  const bool hasTruthClusterT0ZBranch = HasBranch(events, "truth_cluster_t0_z");

  TTreeReaderArray<float> nodeY(reader, "node_y");
  TTreeReaderArray<float> nodeZ(reader, "node_z");
  TTreeReaderArray<float> nodeT(reader, "node_t");
  TTreeReaderArray<float> nodeEnergy(reader, "node_energy");
  TTreeReaderArray<float> nodeProb(reader, "node_prob");
  TTreeReaderArray<float> nodeLabel(reader, "node_label");
  TTreeReaderArray<unsigned char> nodeValid(reader, "node_valid");

  std::unique_ptr<TTreeReaderArray<long long>> nodeTruthCluster;
  if (hasNodeTruthClusterBranch) nodeTruthCluster.reset(new TTreeReaderArray<long long>(reader, "node_truth_cluster"));

  std::unique_ptr<TTreeReaderArray<float>> nodeTruthTime;
  if (hasNodeTruthTimeBranch) nodeTruthTime.reset(new TTreeReaderArray<float>(reader, "node_truth_time"));
  std::unique_ptr<TTreeReaderArray<float>> nodeTruthEnergy;
  if (hasNodeTruthEnergyBranch) nodeTruthEnergy.reset(new TTreeReaderArray<float>(reader, "node_truth_energy"));

  // These reco-pulse quantities are only copied to the output tree for truth-mode
  // diagnostics.  They are not used for t0/energy residuals unless the main
  // node_t/node_energy branches already contain reco quantities.
  std::unique_ptr<TTreeReaderArray<float>> nodeRecoT;
  if (hasNodeRecoTBranch) nodeRecoT.reset(new TTreeReaderArray<float>(reader, "node_reco_t"));
  std::unique_ptr<TTreeReaderArray<float>> nodeRecoAmp;
  if (hasNodeRecoAmpBranch) nodeRecoAmp.reset(new TTreeReaderArray<float>(reader, "node_reco_amp"));
  std::unique_ptr<TTreeReaderArray<float>> nodeRecoArea;
  if (hasNodeRecoAreaBranch) nodeRecoArea.reset(new TTreeReaderArray<float>(reader, "node_reco_area"));

  TTreeReaderArray<int> edgeSrc(reader, "edge_src");
  TTreeReaderArray<int> edgeDst(reader, "edge_dst");
  TTreeReaderArray<float> edgeProb(reader, "edge_prob");
  TTreeReaderArray<float> edgeLabel(reader, "edge_label");
  TTreeReaderArray<unsigned char> edgeValid(reader, "edge_valid");

  std::unique_ptr<TTreeReaderArray<long long>> truthClusterId;
  if (hasTruthClusterIdBranch) truthClusterId.reset(new TTreeReaderArray<long long>(reader, "truth_cluster_id"));
  std::unique_ptr<TTreeReaderArray<long long>> truthClusterPrimaryId;
  if (hasTruthClusterPrimaryIdBranch) truthClusterPrimaryId.reset(
      new TTreeReaderArray<long long>(reader, "truth_cluster_primary_id"));
  std::unique_ptr<TTreeReaderArray<long long>> truthClusterTimeSlice;
  if (hasTruthClusterTimeSliceBranch) truthClusterTimeSlice.reset(
      new TTreeReaderArray<long long>(reader, "truth_cluster_time_slice"));
  std::unique_ptr<TTreeReaderArray<long long>> truthClusterSize;
  if (hasTruthClusterSizeBranch) truthClusterSize.reset(new TTreeReaderArray<long long>(reader, "truth_cluster_size"));
  std::unique_ptr<TTreeReaderArray<float>> truthClusterEnergy;
  if (hasTruthClusterEnergyBranch) truthClusterEnergy.reset(new TTreeReaderArray<float>(reader, "truth_cluster_energy"));
  std::unique_ptr<TTreeReaderArray<float>> truthClusterT0;
  if (hasTruthClusterT0Branch) truthClusterT0.reset(new TTreeReaderArray<float>(reader, "truth_cluster_t0"));
  std::unique_ptr<TTreeReaderArray<float>> truthClusterT0X;
  if (hasTruthClusterT0XBranch) truthClusterT0X.reset(
      new TTreeReaderArray<float>(reader, "truth_cluster_t0_x"));
  std::unique_ptr<TTreeReaderArray<float>> truthClusterT0Y;
  if (hasTruthClusterT0YBranch) truthClusterT0Y.reset(new TTreeReaderArray<float>(reader, "truth_cluster_t0_y"));
  std::unique_ptr<TTreeReaderArray<float>> truthClusterT0Z;
  if (hasTruthClusterT0ZBranch) truthClusterT0Z.reset(new TTreeReaderArray<float>(reader, "truth_cluster_t0_z"));

  std::cout << "[config] optional truth-mode branches:\n"
            << "  graph_source       : " << (hasGraphSourceBranch ? "yes" : "no") << '\n'
            << "  node_truth_cluster : " << (hasNodeTruthClusterBranch ? "yes" : "no") << '\n'
            << "  node_truth_time    : " << (hasNodeTruthTimeBranch ? "yes" : "no") << '\n'
            << "  node_truth_energy  : " << (hasNodeTruthEnergyBranch ? "yes" : "no") << '\n'
            << "  node_reco_t/amp/area: " << ((hasNodeRecoTBranch || hasNodeRecoAmpBranch || hasNodeRecoAreaBranch) ? "yes" : "no") << '\n'
            << "  truth_cluster_*    : " << (hasTruthClusterIdBranch ? "yes" : "no") << '\n'
            << "  truth primary/slice: "
            << ((hasTruthClusterPrimaryIdBranch && hasTruthClusterTimeSliceBranch) ? "yes" : "no")
            << std::endl;

  TString actualOutputRoot = ResolveOutputRoot(inputRoot, outputRoot);
  TString actualOutputDir = DirName(actualOutputRoot.Data());
  gSystem->mkdir(actualOutputDir, kTRUE);

  std::unique_ptr<TFile> fout(TFile::Open(actualOutputRoot, "RECREATE"));
  if (!fout || fout->IsZombie()) {
    std::cerr << "[error] Cannot create output ROOT file: " << actualOutputRoot << std::endl;
    return;
  }

  TTree outTree("clusters", "ECAL clusters reconstructed from GNN evaluation ROOT file");

  long long outEventIndex = -1;
  int outFileIndex = -1;
  long long outGraphIndexInFile = -1;
  int outGraphSource = -1;
  int outUsingTruthNodeObservables = 0;
  int outNNode = 0;
  int outNRecoClusters = 0;
  int outNTruthClusters = 0;
  int outNMatchedTruthClusters = 0;

  std::vector<int> recoSize;
  std::vector<float> recoMeanY;
  std::vector<float> recoMeanZ;
  std::vector<float> recoMeanNodeProb;
  std::vector<float> recoT0;
  std::vector<float> recoT0Y;
  std::vector<float> recoT0Z;
  std::vector<float> recoEnergy;
  std::vector<long long> recoBestTruthId;
  std::vector<float> recoBestTruthPurity;

  // Present only when evaluate_ecal_gnn.py wrote truth-mode diagnostic branches.
  // They are copied per reconstructed cluster, using the same node-probability
  // weighted mean for time and amplitude/area sum over the selected reco cluster.
  std::vector<float> recoMeanRecoT;
  std::vector<float> recoSumRecoAmp;
  std::vector<float> recoSumRecoArea;

  std::vector<long long> truthId;
  std::vector<long long> truthPrimaryId;
  std::vector<long long> truthTimeSlice;
  std::vector<int> truthWindowSize;
  std::vector<int> truthFullSize;
  std::vector<unsigned char> truthMatched;
  std::vector<float> truthBestCompleteness;
  std::vector<float> truthBestPurity;
  std::vector<int> truthBestRecoIndex;
  std::vector<float> truthT0;
  std::vector<float> truthT0X;
  std::vector<float> truthT0Y;
  std::vector<float> truthT0Z;
  std::vector<float> truthEnergy;
  std::vector<float> truthBestRecoT0;
  std::vector<float> truthBestRecoT0Y;
  std::vector<float> truthBestRecoT0Z;
  std::vector<float> truthBestRecoEnergy;
  std::vector<float> truthDeltaT0;
  std::vector<float> truthDeltaT0Y;
  std::vector<float> truthDeltaT0Z;
  std::vector<float> truthDeltaT0R;
  std::vector<float> truthEnergyResidual;
  std::vector<float> truthEnergyResponse;
  std::vector<int> truthSplitFragments;
  std::vector<unsigned char> truthSplit;

  outTree.Branch("event_index", &outEventIndex);
  outTree.Branch("file_index", &outFileIndex);
  outTree.Branch("graph_index_in_file", &outGraphIndexInFile);
  outTree.Branch("graph_source", &outGraphSource);
  outTree.Branch("using_truth_node_observables", &outUsingTruthNodeObservables);
  outTree.Branch("n_node", &outNNode);
  outTree.Branch("n_reco_clusters", &outNRecoClusters);
  outTree.Branch("n_truth_clusters", &outNTruthClusters);
  outTree.Branch("n_matched_truth_clusters", &outNMatchedTruthClusters);
  outTree.Branch("reco_size", &recoSize);
  outTree.Branch("reco_mean_y", &recoMeanY);
  outTree.Branch("reco_mean_z", &recoMeanZ);
  outTree.Branch("reco_mean_node_prob", &recoMeanNodeProb);
  outTree.Branch("reco_t0", &recoT0);
  outTree.Branch("reco_t0_y", &recoT0Y);
  outTree.Branch("reco_t0_z", &recoT0Z);
  outTree.Branch("reco_energy", &recoEnergy);
  outTree.Branch("reco_best_truth_id", &recoBestTruthId);
  outTree.Branch("reco_best_truth_purity", &recoBestTruthPurity);
  outTree.Branch("reco_mean_reco_t", &recoMeanRecoT);
  outTree.Branch("reco_sum_reco_amp", &recoSumRecoAmp);
  outTree.Branch("reco_sum_reco_area", &recoSumRecoArea);
  outTree.Branch("truth_id", &truthId);
  outTree.Branch("truth_primary_id", &truthPrimaryId);
  outTree.Branch("truth_time_slice", &truthTimeSlice);
  outTree.Branch("truth_window_size", &truthWindowSize);
  outTree.Branch("truth_full_size", &truthFullSize);
  outTree.Branch("truth_matched", &truthMatched);
  outTree.Branch("truth_best_completeness", &truthBestCompleteness);
  outTree.Branch("truth_best_purity", &truthBestPurity);
  outTree.Branch("truth_best_reco_index", &truthBestRecoIndex);
  outTree.Branch("truth_t0", &truthT0);
  outTree.Branch("truth_t0_x", &truthT0X);
  outTree.Branch("truth_t0_y", &truthT0Y);
  outTree.Branch("truth_t0_z", &truthT0Z);
  outTree.Branch("truth_energy", &truthEnergy);
  outTree.Branch("truth_best_reco_t0", &truthBestRecoT0);
  outTree.Branch("truth_best_reco_t0_y", &truthBestRecoT0Y);
  outTree.Branch("truth_best_reco_t0_z", &truthBestRecoT0Z);
  outTree.Branch("truth_best_reco_energy", &truthBestRecoEnergy);
  outTree.Branch("truth_delta_t0", &truthDeltaT0);
  outTree.Branch("truth_delta_t0_y", &truthDeltaT0Y);
  outTree.Branch("truth_delta_t0_z", &truthDeltaT0Z);
  outTree.Branch("truth_delta_t0_r", &truthDeltaT0R);
  outTree.Branch("truth_energy_residual", &truthEnergyResidual);
  outTree.Branch("truth_energy_response", &truthEnergyResponse);
  outTree.Branch("truth_split_fragments", &truthSplitFragments);
  outTree.Branch("truth_split", &truthSplit);

  TH1D hNNode("h_n_node", "Nodes per seed-window graph;n_{node};graphs", 100, 0, 1000);
  TH1D hNTruth("h_n_truth_clusters", "Truth clusters per seed-window graph;n_{truth clusters};graphs", 30, -0.5, 29.5);
  TH1D hNReco("h_n_reco_clusters", "Reco clusters per seed-window graph;n_{reco clusters};graphs", 30, -0.5, 29.5);
  TH1D hNMatched("h_n_matched_truth_clusters", "Matched truth clusters per graph;n_{matched truth clusters};graphs", 30, -0.5, 29.5);
  TH2D hNRecoVsTruth("h_n_reco_vs_truth", "Reco vs truth clusters per graph;n_{truth clusters};n_{reco clusters};graphs", 20, -0.5, 19.5, 20, -0.5, 19.5);

  TH1D hTruthSize("h_truth_window_size", "Truth cluster nodes inside seed window;truth nodes in window;clusters", 100, 0.5, 100.5);
  TH1D hRecoSize("h_reco_size", "Reco cluster size;reco cluster size [nodes];clusters", 100, 0.5, 100.5);
  TH1D hTruthBestCompleteness("h_truth_best_completeness", "Best completeness for each truth cluster in window;best completeness;truth clusters", 110, 0.0, 1.1);
  TH1D hTruthBestPurity("h_truth_best_purity", "Purity of best-matching reco;best purity;truth clusters", 110, 0.0, 1.1);
  TH1D hTruthBestCompletenessMatched("h_truth_best_completeness_matched", "Completeness for matched truth clusters;completeness;matched truth clusters", 110, 0.0, 1.1);
  TH1D hTruthBestPurityMatched("h_truth_best_purity_matched", "Purity for matched truth clusters;purity;matched truth clusters", 110, 0.0, 1.1);
  TH1D hRecoBestPurity("h_reco_best_truth_purity", "Best truth purity for each reco cluster;best truth purity;reco clusters", 110, 0.0, 1.1);
  TH1D hRecoMeanNodeProb("h_reco_mean_node_prob", "Mean node probability in reco cluster;mean node probability;reco clusters", 110, 0.0, 1.1);
  TH2D hCompletenessVsPurity("h_completeness_vs_purity", "Truth-cluster match quality;best purity;best completeness;truth clusters", 110, 0.0, 1.1, 110, 0.0, 1.1);
  TH2D hRecoPurityVsSize("h_reco_purity_vs_size", "Reco purity vs reco size;reco size [nodes];best truth purity;reco clusters", 100, 0.5, 100.5, 110, 0.0, 1.1);

  TH1D hT0Residual("h_t0_residual", "Cluster t0 resolution;t0_{reco} - t0_{truth} [ns];matched truth clusters", 100, -50.0, 50.0);
  TH2D hT0RecoVsTruth("h_t0_reco_vs_truth", "Reco vs truth cluster t0;t0_{truth} [ns];t0_{reco} [ns];matched truth clusters", 200, -100.0, 100.0, 200, -100.0, 100.0);
  TH1D hT0PosResidualY("h_t0_pos_residual_y", "Cluster t0-position residual;#Delta y [mm];matched truth clusters", 20, -200.0, 200.0);
  TH1D hT0PosResidualZ("h_t0_pos_residual_z", "Cluster t0-position residual;#Delta z [mm];matched truth clusters", 20, -200.0, 200.0);
  TH1D hT0PosResidualR("h_t0_pos_residual_r", "Cluster t0-position radial residual;#sqrt{#Delta y^{2}+#Delta z^{2}} [mm];matched truth clusters", 10, 0.0, 200.0);
  TH2D hT0PosResidual2D("h_t0_pos_residual2D", "Cluster t0-position residual;#Delta y [mm];#Delta z [mm];matched truth clusters", 21, -210.0, 210.0, 21, -210.0, 210.0);
  TH1D hEnergyResidual("h_energy_residual", "Cluster energy residual;E_{reco} - E_{truth} [MeV or MeV-equivalent];matched truth clusters", 200, -100.0, 100.0);
  TH1D hEnergyResponse("h_energy_response", "Cluster energy resolution;(E_{reco} - E_{truth})/E_{truth};matched truth clusters", 200, -2.0, 2.0);
  TH2D hEnergyRecoVsTruth("h_energy_reco_vs_truth", "Reco vs truth cluster energy;E_{truth} [MeV or MeV-equivalent];E_{reco} [MeV or MeV-equivalent];matched truth clusters", 120, 0.0, 120.0, 120, 0.0, 120.0);
  TH1D hTruthSplitFragments("h_truth_split_fragments", "Reco fragments per truth cluster inside one seed window;n_{reco fragments overlapping truth};truth clusters", 20, -0.5, 19.5);

  TH1D hNodeProbSignal("h_node_prob_signal", "Node GNN score;node probability;nodes", 110, 0.0, 1.1);
  TH1D hNodeProbNoise("h_node_prob_noise", "Node GNN score;node probability;nodes", 110, 0.0, 1.1);
  TH1D hEdgeProbSame("h_edge_prob_same", "Edge GNN score;edge probability;edges", 110, 0.0, 1.1);
  TH1D hEdgeProbDiff("h_edge_prob_diff", "Edge GNN score;edge probability;edges", 110, 0.0, 1.1);

  TEfficiency hEffTruthVsSize("h_eff_truth_vs_size", "Truth cluster efficiency vs window size;truth nodes in window;efficiency", 100, 0.5, 100.5);
  TEfficiency hFakeRecoVsSize("h_fake_reco_vs_size", "Fake reco fraction vs reco size;reco cluster size [nodes];fake fraction", 100, 0.5, 100.5);
  TEfficiency hSplitRateVsSize("h_splitting_rate_vs_size", "Splitting rate vs truth size inside seed window;truth nodes in window;splitting rate", 100, 0.5, 100.5);

  long long totalEvents = 0;
  long long totalRecoClusters = 0;
  long long totalTruthClusters = 0;
  long long totalMatchedTruthClusters = 0;
  long long totalFakeRecoClusters = 0;
  long long totalSplitTruthClusters = 0;
  long long totalTruthObservableGraphs = 0;

  while (reader.Next()) {
    const int nNode = static_cast<int>(*nNodeBranch);

    const auto vNodeY = CopyArray(nodeY);
    const auto vNodeZ = CopyArray(nodeZ);
    const auto vNodeTBase = CopyArray(nodeT);
    const auto vNodeEnergyBase = CopyArray(nodeEnergy);
    const auto vNodeProb = CopyArray(nodeProb);
    const auto vNodeLabel = CopyArray(nodeLabel);
    const auto vNodeValid = CopyArray(nodeValid);
    const auto vNodeTruthCluster = CopyArrayIfPresent(nodeTruthCluster);
    const auto vNodeTruthT = CopyArrayIfPresent(nodeTruthTime);
    const auto vNodeTruthEnergy = CopyArrayIfPresent(nodeTruthEnergy);
    const auto vNodeRecoT = CopyArrayIfPresent(nodeRecoT);
    const auto vNodeRecoAmp = CopyArrayIfPresent(nodeRecoAmp);
    const auto vNodeRecoArea = CopyArrayIfPresent(nodeRecoArea);

    const int inputGraphSource =
        graphSourceBranch ? static_cast<int>(**graphSourceBranch) : -1;
    const bool useTruthNodeObservables =
        inputGraphSource == 1
        && HasNodeVectorSize(vNodeTruthT, nNode)
        && HasNodeVectorSize(vNodeTruthEnergy, nNode);
    const std::vector<float>& vNodeT = useTruthNodeObservables ? vNodeTruthT : vNodeTBase;
    const std::vector<float>& vNodeEnergy = useTruthNodeObservables ? vNodeTruthEnergy : vNodeEnergyBase;
    if (useTruthNodeObservables) ++totalTruthObservableGraphs;

    const auto vEdgeSrc = CopyArray(edgeSrc);
    const auto vEdgeDst = CopyArray(edgeDst);
    const auto vEdgeProb = CopyArray(edgeProb);
    const auto vEdgeLabel = CopyArray(edgeLabel);
    const auto vEdgeValid = CopyArray(edgeValid);

    const auto vTruthClusterId = CopyArrayIfPresent(truthClusterId);
    const auto vTruthClusterPrimaryId = CopyArrayIfPresent(truthClusterPrimaryId);
    const auto vTruthClusterTimeSlice = CopyArrayIfPresent(truthClusterTimeSlice);
    const auto vTruthClusterSize = CopyArrayIfPresent(truthClusterSize);
    const auto vTruthClusterEnergy = CopyArrayIfPresent(truthClusterEnergy);
    const auto vTruthClusterT0 = CopyArrayIfPresent(truthClusterT0);
    const auto vTruthClusterT0X = CopyArrayIfPresent(truthClusterT0X);
    const auto vTruthClusterT0Y = CopyArrayIfPresent(truthClusterT0Y);
    const auto vTruthClusterT0Z = CopyArrayIfPresent(truthClusterT0Z);

    if (static_cast<int>(vNodeProb.size()) != nNode) {
      std::cerr << "[warn] event " << *eventIndex << " has n_node=" << nNode
                << " but node_prob size=" << vNodeProb.size() << "; skipped" << std::endl;
      continue;
    }
    if (static_cast<int>(vNodeT.size()) != nNode || static_cast<int>(vNodeEnergy.size()) != nNode) {
      std::cerr << "[warn] event " << *eventIndex << " has invalid node observable sizes: "
                << "n_node=" << nNode << " node_t=" << vNodeT.size()
                << " node_energy=" << vNodeEnergy.size() << "; skipped" << std::endl;
      continue;
    }

    hNNode.Fill(nNode);
    for (int i = 0; i < nNode; ++i) {
      if (i >= static_cast<int>(vNodeValid.size()) || vNodeValid[i] == 0) continue;
      if (i >= static_cast<int>(vNodeLabel.size()) || i >= static_cast<int>(vNodeProb.size())) continue;
      if (vNodeLabel[i] > 0.5f) hNodeProbSignal.Fill(vNodeProb[i]);
      else hNodeProbNoise.Fill(vNodeProb[i]);
    }
    for (std::size_t e = 0; e < vEdgeProb.size(); ++e) {
      if (e >= vEdgeValid.size() || vEdgeValid[e] == 0) continue;
      if (e >= vEdgeLabel.size()) continue;
      if (vEdgeLabel[e] > 0.5f) hEdgeProbSame.Fill(vEdgeProb[e]);
      else hEdgeProbDiff.Fill(vEdgeProb[e]);
    }

    auto recoClusters = BuildPredictedClusters(
        nNode, vNodeProb, vNodeValid, vEdgeSrc, vEdgeDst, vEdgeProb, vEdgeValid,
        nodeThreshold, edgeThreshold, minRecoNodes);

    std::vector<long long> allTruthIds;
    auto truthClustersAll = BuildTruthClusters(
        nNode, vNodeLabel, vNodeValid, vNodeTruthCluster, vEdgeSrc, vEdgeDst,
        vEdgeLabel, vEdgeValid, allTruthIds);

    std::vector<std::vector<int>> truthClusters;
    truthId.clear();
    for (std::size_t i = 0; i < truthClustersAll.size(); ++i) {
      if (static_cast<int>(truthClustersAll[i].size()) < minTruthNodes) continue;
      truthClusters.push_back(std::move(truthClustersAll[i]));
      truthId.push_back(allTruthIds[i]);
    }

    const auto recoObs = ComputeRecoObservableList(recoClusters, vNodeY, vNodeZ, vNodeT, vNodeEnergy);
    const auto truthObsMap = BuildTruthObservableMap(
        vTruthClusterId, vTruthClusterEnergy, vTruthClusterT0, vTruthClusterT0X,
        vTruthClusterT0Y, vTruthClusterT0Z);
    const auto truthSizeMap = BuildTruthSizeMap(vTruthClusterId, vTruthClusterSize);
    const auto truthPrimaryMap = BuildTruthIntegerMap(
        vTruthClusterId, vTruthClusterPrimaryId);
    const auto truthTimeSliceMap = BuildTruthIntegerMap(
        vTruthClusterId, vTruthClusterTimeSlice);

    truthPrimaryId.clear();
    truthTimeSlice.clear();
    truthWindowSize.clear();
    truthFullSize.clear();
    truthMatched.clear();
    truthBestCompleteness.clear();
    truthBestPurity.clear();
    truthBestRecoIndex.clear();
    truthT0.clear();
    truthT0X.clear();
    truthT0Y.clear();
    truthT0Z.clear();
    truthEnergy.clear();
    truthBestRecoT0.clear();
    truthBestRecoT0Y.clear();
    truthBestRecoT0Z.clear();
    truthBestRecoEnergy.clear();
    truthDeltaT0.clear();
    truthDeltaT0Y.clear();
    truthDeltaT0Z.clear();
    truthDeltaT0R.clear();
    truthEnergyResidual.clear();
    truthEnergyResponse.clear();
    truthSplitFragments.clear();
    truthSplit.clear();

    int matchedTruthThisEvent = 0;
    int splitTruthThisEvent = 0;
    for (std::size_t it = 0; it < truthClusters.size(); ++it) {
      const auto& truth = truthClusters[it];
      const long long tid = truthId[it];
      float bestCompleteness = 0.0f;
      float bestPurity = 0.0f;
      int bestRecoIndex = -1;
      int splitFragments = 0;

      for (std::size_t ir = 0; ir < recoClusters.size(); ++ir) {
        const auto& reco = recoClusters[ir];
        const int overlap = CountOverlap(truth, reco);
        if (overlap <= 0) continue;
        const int windowSize = static_cast<int>(truth.size());
        auto sizeIt = truthSizeMap.find(tid);
        const int fullSize = (sizeIt != truthSizeMap.end()) ? static_cast<int>(sizeIt->second) : windowSize;
        const float windowCompleteness = windowSize > 0 ? static_cast<float>(overlap) / static_cast<float>(windowSize) : 0.0f;
        const float windowCoverage = fullSize > 0 ? static_cast<float>(windowSize) / static_cast<float>(fullSize) : 0.0f;
        const float fullCompleteness = fullSize > 0 ? static_cast<float>(overlap) / static_cast<float>(fullSize) : 0.0f;
        //const float completeness = fullCompleteness;
        //const float completeness = windowCoverage;
        const float completeness = windowCompleteness;
        const float purity = static_cast<float>(overlap) / static_cast<float>(reco.size());
        if (overlap >= minSplitOverlapNodes && completeness >= minSplitCompleteness) ++splitFragments;
        if (completeness > bestCompleteness || (completeness == bestCompleteness && purity > bestPurity)) {
          bestCompleteness = completeness;
          bestPurity = purity;
          bestRecoIndex = static_cast<int>(ir);
        }
      }

      const bool matched = (bestCompleteness >= minCompleteness && bestPurity >= minPurity && bestRecoIndex >= 0);
      const bool split = (splitFragments >= 2);
      if (matched) ++matchedTruthThisEvent;
      if (split) ++splitTruthThisEvent;

      ClusterObservables tobs;
      auto foundTruthObs = truthObsMap.find(tid);
      if (foundTruthObs != truthObsMap.end()) {
        tobs = foundTruthObs->second;
      } else {
        // Robust fallback for older ROOT files: use truth nodes visible in this window.
        tobs = ComputeRecoObservables(truth, vNodeY, vNodeZ, vNodeT, vNodeEnergy);
      }

      const float nan = std::numeric_limits<float>::quiet_NaN();
      float recoT0ForTruth = nan;
      float recoYForTruth = nan;
      float recoZForTruth = nan;
      float recoEnergyForTruth = nan;
      float deltaT0 = nan;
      float deltaY = nan;
      float deltaZ = nan;
      float deltaR = nan;
      float energyResidual = nan;
      float energyResponse = nan;

      if (bestRecoIndex >= 0 && bestRecoIndex < static_cast<int>(recoObs.size())) {
        const ClusterObservables& robs = recoObs[bestRecoIndex];
        recoT0ForTruth = robs.t0;
        recoYForTruth = robs.t0Y;
        recoZForTruth = robs.t0Z;
        recoEnergyForTruth = robs.energy;
        deltaT0 = robs.t0 - tobs.t0;
        deltaY = robs.t0Y - tobs.t0Y;
        deltaZ = robs.t0Z - tobs.t0Z;
        deltaR = std::sqrt(deltaY * deltaY + deltaZ * deltaZ);
        energyResidual = robs.energy - tobs.energy;
        if (std::isfinite(tobs.energy) && std::abs(tobs.energy) > 1.0e-12f) {
          energyResponse = energyResidual / tobs.energy;
        }
      }

      auto sizeIt = truthSizeMap.find(tid);
      const int fullSize = (sizeIt != truthSizeMap.end()) ? static_cast<int>(sizeIt->second) : static_cast<int>(truth.size());

      auto primaryIt = truthPrimaryMap.find(tid);
      auto sliceIt = truthTimeSliceMap.find(tid);
      truthPrimaryId.push_back(
          primaryIt != truthPrimaryMap.end() ? primaryIt->second : -1);
      truthTimeSlice.push_back(
          sliceIt != truthTimeSliceMap.end() ? sliceIt->second : -1);
      truthWindowSize.push_back(static_cast<int>(truth.size()));
      truthFullSize.push_back(fullSize);
      truthMatched.push_back(matched ? 1 : 0);
      truthBestCompleteness.push_back(bestCompleteness);
      truthBestPurity.push_back(bestPurity);
      truthBestRecoIndex.push_back(bestRecoIndex);
      truthT0.push_back(tobs.t0);
      truthT0X.push_back(tobs.t0X);
      truthT0Y.push_back(tobs.t0Y);
      truthT0Z.push_back(tobs.t0Z);
      truthEnergy.push_back(tobs.energy);
      truthBestRecoT0.push_back(recoT0ForTruth);
      truthBestRecoT0Y.push_back(recoYForTruth);
      truthBestRecoT0Z.push_back(recoZForTruth);
      truthBestRecoEnergy.push_back(recoEnergyForTruth);
      truthDeltaT0.push_back(deltaT0);
      truthDeltaT0Y.push_back(deltaY);
      truthDeltaT0Z.push_back(deltaZ);
      truthDeltaT0R.push_back(deltaR);
      truthEnergyResidual.push_back(energyResidual);
      truthEnergyResponse.push_back(energyResponse);
      truthSplitFragments.push_back(splitFragments);
      truthSplit.push_back(split ? 1 : 0);

      hTruthSize.Fill(static_cast<double>(truth.size()));
      hTruthBestCompleteness.Fill(bestCompleteness);
      hTruthBestPurity.Fill(bestPurity);
      hCompletenessVsPurity.Fill(bestPurity, bestCompleteness);
      hEffTruthVsSize.Fill(matched, static_cast<double>(truth.size()));
      hSplitRateVsSize.Fill(split, static_cast<double>(truth.size()));
      hTruthSplitFragments.Fill(splitFragments);
      if (matched) {
        hTruthBestCompletenessMatched.Fill(bestCompleteness);
        hTruthBestPurityMatched.Fill(bestPurity);
        if (std::isfinite(deltaT0)) hT0Residual.Fill(deltaT0);
        if (std::isfinite(tobs.t0) && std::isfinite(recoT0ForTruth)) hT0RecoVsTruth.Fill(tobs.t0, recoT0ForTruth);
        if (std::isfinite(deltaY)) hT0PosResidualY.Fill(deltaY);
        if (std::isfinite(deltaZ)) hT0PosResidualZ.Fill(deltaZ);
        if (std::isfinite(deltaR)) hT0PosResidualR.Fill(deltaR);
        if (std::isfinite(deltaY) && std::isfinite(deltaZ)) hT0PosResidual2D.Fill(deltaY, deltaZ);
        if (std::isfinite(energyResidual)) hEnergyResidual.Fill(energyResidual);
        if (std::isfinite(energyResponse)) hEnergyResponse.Fill(energyResponse);
        if (std::isfinite(tobs.energy) && std::isfinite(recoEnergyForTruth)) hEnergyRecoVsTruth.Fill(tobs.energy, recoEnergyForTruth);
      }
    }

    recoSize.clear();
    recoMeanY.clear();
    recoMeanZ.clear();
    recoMeanNodeProb.clear();
    recoT0.clear();
    recoT0Y.clear();
    recoT0Z.clear();
    recoEnergy.clear();
    recoBestTruthId.clear();
    recoBestTruthPurity.clear();
    recoMeanRecoT.clear();
    recoSumRecoAmp.clear();
    recoSumRecoArea.clear();

    int fakeRecoThisEvent = 0;
    for (std::size_t ir = 0; ir < recoClusters.size(); ++ir) {
      const auto& reco = recoClusters[ir];
      const ClusterObservables& robs = recoObs[ir];
      double sumW = 0.0;
      double sumY = 0.0;
      double sumZ = 0.0;
      double sumProb = 0.0;
      double sumRecoTWeighted = 0.0;
      double sumRecoAmp = 0.0;
      double sumRecoArea = 0.0;
      for (const int iNode : reco) {
        const double w = std::max(1.0e-6, static_cast<double>(vNodeProb[iNode]));
        sumW += w;
        if (iNode < static_cast<int>(vNodeY.size())) sumY += w * vNodeY[iNode];
        if (iNode < static_cast<int>(vNodeZ.size())) sumZ += w * vNodeZ[iNode];
        if (iNode < static_cast<int>(vNodeRecoT.size()) && std::isfinite(vNodeRecoT[iNode])) sumRecoTWeighted += w * vNodeRecoT[iNode];
        if (iNode < static_cast<int>(vNodeRecoAmp.size()) && std::isfinite(vNodeRecoAmp[iNode])) sumRecoAmp += vNodeRecoAmp[iNode];
        if (iNode < static_cast<int>(vNodeRecoArea.size()) && std::isfinite(vNodeRecoArea[iNode])) sumRecoArea += vNodeRecoArea[iNode];
        sumProb += vNodeProb[iNode];
      }

      long long bestTruthId = -1;
      float bestPurity = 0.0f;
      for (std::size_t it = 0; it < truthClusters.size(); ++it) {
        const int overlap = CountOverlap(reco, truthClusters[it]);
        if (overlap <= 0) continue;
        const float purity = static_cast<float>(overlap) / static_cast<float>(reco.size());
        if (purity > bestPurity) {
          bestPurity = purity;
          bestTruthId = truthId[it];
        }
      }
      const bool fakeReco = (bestPurity < minPurity);
      if (fakeReco) ++fakeRecoThisEvent;

      recoSize.push_back(static_cast<int>(reco.size()));
      recoMeanY.push_back(sumW > 0.0 ? static_cast<float>(sumY / sumW) : 0.0f);
      recoMeanZ.push_back(sumW > 0.0 ? static_cast<float>(sumZ / sumW) : 0.0f);
      recoMeanNodeProb.push_back(!reco.empty() ? static_cast<float>(sumProb / reco.size()) : 0.0f);
      recoT0.push_back(robs.t0);
      recoT0Y.push_back(robs.t0Y);
      recoT0Z.push_back(robs.t0Z);
      recoEnergy.push_back(robs.energy);
      recoBestTruthId.push_back(bestTruthId);
      recoBestTruthPurity.push_back(bestPurity);
      recoMeanRecoT.push_back(!vNodeRecoT.empty() && sumW > 0.0 ? static_cast<float>(sumRecoTWeighted / sumW) : std::numeric_limits<float>::quiet_NaN());
      recoSumRecoAmp.push_back(!vNodeRecoAmp.empty() ? static_cast<float>(sumRecoAmp) : std::numeric_limits<float>::quiet_NaN());
      recoSumRecoArea.push_back(!vNodeRecoArea.empty() ? static_cast<float>(sumRecoArea) : std::numeric_limits<float>::quiet_NaN());

      hRecoSize.Fill(static_cast<double>(reco.size()));
      hRecoBestPurity.Fill(bestPurity);
      hRecoMeanNodeProb.Fill(!reco.empty() ? static_cast<float>(sumProb / reco.size()) : 0.0f);
      hRecoPurityVsSize.Fill(static_cast<double>(reco.size()), bestPurity);
      hFakeRecoVsSize.Fill(fakeReco, static_cast<double>(reco.size()));
    }

    outEventIndex = *eventIndex;
    outFileIndex = *fileIndex;
    outGraphIndexInFile = *graphIndexInFile;
    // Avoid a hard dependency on the exact integer type of graph_source in the
    // eval ROOT file.  Convention: 1 = truth-feature graph, 0 = reco-feature graph,
    // -1 = branch absent/unknown.
    outGraphSource = inputGraphSource;
    outUsingTruthNodeObservables = useTruthNodeObservables ? 1 : 0;
    outNNode = nNode;
    outNRecoClusters = static_cast<int>(recoClusters.size());
    outNTruthClusters = static_cast<int>(truthClusters.size());
    outNMatchedTruthClusters = matchedTruthThisEvent;
    outTree.Fill();

    hNTruth.Fill(outNTruthClusters);
    hNReco.Fill(outNRecoClusters);
    hNMatched.Fill(outNMatchedTruthClusters);
    hNRecoVsTruth.Fill(outNTruthClusters, outNRecoClusters);

    ++totalEvents;
    totalRecoClusters += static_cast<long long>(recoClusters.size());
    totalTruthClusters += static_cast<long long>(truthClusters.size());
    totalMatchedTruthClusters += matchedTruthThisEvent;
    totalFakeRecoClusters += fakeRecoThisEvent;
    totalSplitTruthClusters += splitTruthThisEvent;
  }

  const double truthEfficiency = totalTruthClusters > 0 ? static_cast<double>(totalMatchedTruthClusters) / totalTruthClusters : 0.0;
  const double fakeRecoRate = totalRecoClusters > 0 ? static_cast<double>(totalFakeRecoClusters) / totalRecoClusters : 0.0;
  const double splittingRate = totalTruthClusters > 0 ? static_cast<double>(totalSplitTruthClusters) / totalTruthClusters : 0.0;

  fout->cd();
  outTree.Write();
  hNNode.Write();
  hNTruth.Write();
  hNReco.Write();
  hNMatched.Write();
  hNRecoVsTruth.Write();
  hTruthSize.Write();
  hRecoSize.Write();
  hTruthBestCompleteness.Write();
  hTruthBestPurity.Write();
  hTruthBestCompletenessMatched.Write();
  hTruthBestPurityMatched.Write();
  hRecoBestPurity.Write();
  hRecoMeanNodeProb.Write();
  hCompletenessVsPurity.Write();
  hRecoPurityVsSize.Write();
  hT0Residual.Write();
  hT0RecoVsTruth.Write();
  hT0PosResidualY.Write();
  hT0PosResidualZ.Write();
  hT0PosResidualR.Write();
  hT0PosResidual2D.Write();
  hEnergyResidual.Write();
  hEnergyResponse.Write();
  hEnergyRecoVsTruth.Write();
  hTruthSplitFragments.Write();
  hNodeProbSignal.Write();
  hNodeProbNoise.Write();
  hEdgeProbSame.Write();
  hEdgeProbDiff.Write();
  hEffTruthVsSize.Write();
  hFakeRecoVsSize.Write();
  hSplitRateVsSize.Write();

  TString actualPlotDir(plotDir);
  if (actualPlotDir.IsNull()) actualPlotDir = DefaultPlotDir(actualOutputRoot.Data());
  gSystem->mkdir(actualPlotDir, kTRUE);

  DrawOverlayNormalized(hNTruth, "truth", hNReco, "reco", actualPlotDir, "cluster_count_per_seed_window", false);
  DrawOverlayNormalized(hTruthSize, "truth in window", hRecoSize, "reco", actualPlotDir, "cluster_size", true);
  DrawOverlayNormalized(hTruthBestCompleteness, "completeness", hTruthBestPurity, "purity", actualPlotDir, "truth_match_quality", false);
  DrawOverlayNormalized(hNodeProbSignal, "true signal nodes", hNodeProbNoise, "true noise nodes", actualPlotDir, "node_score_separation", true);
  DrawOverlayNormalized(hEdgeProbSame, "true same-cluster edges", hEdgeProbDiff, "true different/noise edges", actualPlotDir, "edge_score_separation", true);

  {
    TCanvas c("truth_efficiency_vs_size", "truth_efficiency_vs_size", 900, 700);
    c.SetGrid();
    hEffTruthVsSize.Draw("AP");
    SaveCanvas(c, actualPlotDir, "truth_efficiency_vs_size");
  }
  {
    TCanvas c("fake_reco_fraction_vs_size", "fake_reco_fraction_vs_size", 900, 700);
    c.SetGrid();
    hFakeRecoVsSize.Draw("AP");
    SaveCanvas(c, actualPlotDir, "fake_reco_fraction_vs_size");
  }
  DrawSingle2D(hCompletenessVsPurity, actualPlotDir, "completeness_vs_purity");
  DrawSingle2D(hNRecoVsTruth, actualPlotDir, "n_reco_vs_truth");
  DrawSingle2D(hRecoPurityVsSize, actualPlotDir, "reco_purity_vs_size");

  DrawSingle(hT0Residual, actualPlotDir, "t0_resolution", false);
  DrawOverlayNormalized(hT0PosResidualY, "#Delta y", hT0PosResidualZ, "#Delta z", actualPlotDir, "t0_position_resolution_components", false);
  DrawSingle(hT0PosResidualR, actualPlotDir, "t0_position_resolution_radial", false);
  DrawSingle2D(hT0PosResidual2D, actualPlotDir, "t0_position_residual_2d");
  DrawSingle(hEnergyResponse, actualPlotDir, "energy_resolution", false);
  DrawSingle(hEnergyResidual, actualPlotDir, "energy_residual", false);
  DrawSingle2D(hEnergyRecoVsTruth, actualPlotDir, "energy_reco_vs_truth");
  DrawSingle(hTruthSplitFragments, actualPlotDir, "truth_split_fragments", false);
  {
    TCanvas c("splitting_rate_vs_truth_size", "splitting_rate_vs_truth_size", 900, 700);
    c.SetGrid();
    hSplitRateVsSize.Draw("AP");
    SaveCanvas(c, actualPlotDir, "splitting_rate_vs_truth_size");
  }

  fout->Close();

  std::cout << "[summary] input ROOT              : " << inputRoot << '\n'
            << "[summary] output ROOT             : " << actualOutputRoot << '\n'
            << "[summary] plot dir                : " << actualPlotDir << '\n'
            << "[summary] seed-window graphs      : " << totalEvents << '\n'
            << "[summary] reco clusters           : " << totalRecoClusters << '\n'
            << "[summary] truth clusters in window: " << totalTruthClusters << '\n'
            << "[summary] matched truth clusters  : " << totalMatchedTruthClusters << '\n'
            << "[summary] truth cluster efficiency: " << truthEfficiency << '\n'
            << "[summary] fake reco cluster rate  : " << fakeRecoRate << '\n'
            << "[summary] splitting rate          : " << splittingRate << '\n'
            << "[summary] graphs using truth node obs: " << totalTruthObservableGraphs << '\n'
            << "[summary] node observable source  : "
            << ((totalTruthObservableGraphs > 0) ? "node_truth_time/node_truth_energy when available" : "node_t/node_energy") << '\n'
            << "[summary] nodeThreshold           : " << nodeThreshold << '\n'
            << "[summary] edgeThreshold           : " << edgeThreshold << '\n'
            << "[summary] minCompleteness         : " << minCompleteness << '\n'
            << "[summary] minPurity               : " << minPurity << '\n'
            << "[summary] minTruthNodes           : " << minTruthNodes << '\n'
            << "[summary] minRecoNodes            : " << minRecoNodes << std::endl;
}
