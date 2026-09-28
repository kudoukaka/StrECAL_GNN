#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <stdexcept>

#include <TRandom3.h>
#include <TVector3.h>

#include <ICOMETEvent.hxx>
#include <IHandle.hxx>
#include <HEPUnits.hxx>
#include <cometEventLoop.hxx>

#include "IG4VHit.hxx"
#include "IG4HitCalo.hxx"
#include "IGeomInfo.hxx"
#include "IG4HitGas.hxx"

class IUserLoop : public COMET::ICOMETEventLoopFunction {
public:
  IUserLoop()
    : fStrawCenterRateMHzPerChannel(0.0),
      fStrawFloorRateMHzPerChannel(0.00),
      fStrawSigmaWire(15.0),
      fStrawReadoutWindowNs(1000.0),
      fStrawElectronMomentumMeV(1.0),
      fStrawEdepMpvMeV(0.003),
      fStrawEdepSigmaMeV(0.001),
      fStrawEdepCapMeV(0.05),
      fNAddedStrawTotal(0),
      fReadoutWindowNs(1024.0),
      fEcalOccupancy(0.01),
      fGammaFraction(0.35),
      fSeed(12345),
      fVerbose(false),
      fNAddedTotal(0),
      fNInvalidEcalGeometry(0)
  {}

  void Usage(void)
  {
    std::cout
      << "Options:\n"
      << "  -O StrawCenterRateMHzPerChannel=0.0\n"
      << "  -O StrawFloorRateMHzPerChannel=0.0\n"
      << "  -O StrawSigmaWire=10.0\n"
      << "  -O StrawReadoutWindowNs=1024\n"
      << "  -O StrawElectronMomentumMeV=1.0\n"
      << "  -O StrawEdepMpvMeV=0.003\n"
      << "  -O StrawEdepSigmaMeV=0.001\n"
      << "  -O StrawEdepCapMeV=0.05\n"
      << "  -O EcalOccupancy=0.01\n"
      << "  -O ReadoutWindowNs=1024\n"
      << "  -O NCrystals=512\n"
      << "  -O GammaFraction=0.6\n"
      << "  -O Seed=12345\n"
      << "  -O Verbose=1\n";
  }

  bool SetOption(std::string opt, std::string val)
  {
    if (opt == "StrawCenterRateMHzPerChannel") {
      fStrawCenterRateMHzPerChannel = std::atof(val.c_str());
      return true;
    }
    if (opt == "StrawFloorRateMHzPerChannel") {
      fStrawFloorRateMHzPerChannel = std::atof(val.c_str());
      return true;
    }
    if (opt == "StrawSigmaWire") {
      fStrawSigmaWire = std::atof(val.c_str());
      return true;
    }
    if (opt == "StrawReadoutWindowNs") {
      fStrawReadoutWindowNs = std::atof(val.c_str());
      return true;
    }
    if (opt == "StrawElectronMomentumMeV") {
      fStrawElectronMomentumMeV = std::atof(val.c_str());
      return true;
    }
    if (opt == "StrawEdepMpvMeV") {
      fStrawEdepMpvMeV = std::atof(val.c_str());
      return true;
    }
    if (opt == "StrawEdepSigmaMeV") {
      fStrawEdepSigmaMeV = std::atof(val.c_str());
      return true;
    }
    if (opt == "StrawEdepCapMeV") {
      fStrawEdepCapMeV = std::atof(val.c_str());
      return true;
    }
    if (opt == "EcalOccupancy") {
      fEcalOccupancy = std::atof(val.c_str());
      return true;
    }
    if (opt == "ReadoutWindowNs") {
      fReadoutWindowNs = std::atof(val.c_str());
      return true;
    }
    if (opt == "GammaFraction") {
      fGammaFraction = std::atof(val.c_str());
      return true;
    }
    if (opt == "Seed") {
      fSeed = static_cast<UInt_t>(std::atoi(val.c_str()));
      return true;
    }
    if (opt == "Verbose") {
      fVerbose = (std::atoi(val.c_str()) != 0);
      return true;
    }

    return true;
  }

  void Initialize(void)
  {
    if (!std::isfinite(fEcalOccupancy) || fEcalOccupancy < 0.0 || fEcalOccupancy >= 1.0
        || !std::isfinite(fStrawCenterRateMHzPerChannel) || fStrawCenterRateMHzPerChannel < 0.0
        || !std::isfinite(fStrawFloorRateMHzPerChannel) || fStrawFloorRateMHzPerChannel < 0.0) {
      throw std::invalid_argument("Noise rates must be finite and nonnegative; EcalOccupancy must be < 1");
    }
    std::cout << "[AddEcalRandomNoise] occupancy=" << fEcalOccupancy
              << " straw_center_rate=" << fStrawCenterRateMHzPerChannel
              << " straw_floor_rate=" << fStrawFloorRateMHzPerChannel << std::endl;
    fRng.SetSeed(fSeed);
  }

  bool operator () (COMET::ICOMETEvent& event)
  {
    COMET::IHandle<COMET::IG4HitContainer> ecalHits = event.Get<COMET::IG4HitContainer>("truth/g4Hits/ECAL");

    if (!ecalHits) {
      if (fVerbose) {
        std::cout << "[AddEcalRandomNoise] no container in event: " << event.GetEventId() << std::endl;
      }
      return true;
    }

    const int nAdded = AddEcalNoiseHits(*ecalHits);

    fNAddedTotal += nAdded;

    if (fVerbose) {
      std::cout << "[AddEcalRandomNoise] event "
                << event.GetEventId()
                << " added " << nAdded
                << " ECAL noise IG4HitCalo hits"
                << std::endl;
    }

    COMET::IHandle<COMET::IG4HitContainer> strawHits = event.Get<COMET::IG4HitContainer>("truth/g4Hits/StrawTrk");
  
    if (strawHits) {
      const int nAddedStraw = AddStrawNoiseHits(*strawHits);
      fNAddedStrawTotal += nAddedStraw;
  
      if (fVerbose) {
        std::cout << "[AddEcalRandomNoise] event "
                  << event.GetEventId()
                  << ": added straw noise hits = "
                  << nAddedStraw
                  << std::endl;
      }
    }

    return true;
  }

  void Finalize(COMET::ICOMETOutput*)
  {
    std::cout << "[AddEcalRandomNoise] total added ECAL noise hits = "
              << fNAddedTotal << std::endl;
    std::cout << "[AddEcalRandomNoise] total added Straw noise hits = "
              << fNAddedStrawTotal
              << std::endl;
    std::cout << "[AddEcalRandomNoise] rejected ECAL noise candidates with invalid geometry = "
              << fNInvalidEcalGeometry << std::endl;
  }

private:
  int AddEcalNoiseHits(COMET::IG4HitContainer& ecalHits)
  {
    if (fEcalOccupancy == 0.0) return 0;
    const auto& ecalInfo = COMET::IGeomInfo::ECAL();
    const UInt_t crystals = ecalInfo.GetNumberOfCrystals();
    const UInt_t blocks = ecalInfo.GetNumberOfBlocks();

    int totalNoise = 0;

    for (int block = 0; block < blocks; block++){
      for(int crystal = 0; crystal < crystals; crystal++){
        const double lambda = -std::log(1.0 - fEcalOccupancy);
        const double fEcalRateMHzPerChannel = lambda / (fReadoutWindowNs * 1.0e-3);
        const int nNoise = fRng.Poisson(lambda);
        for (int i = 0; i < nNoise; ++i) {
          const int seqId = ecalInfo.GetSequenceIdFromECAL(block, crystal);
          if (seqId < 0 || seqId >= static_cast<int>(ecalInfo.GetNumberOfChannels())) {
            ++fNInvalidEcalGeometry;
            continue;
          }
          const int crystalId = ecalInfo.GetCrystalIdFromSequenceId(seqId);

          const double tNs = fRng.Uniform(0, fReadoutWindowNs);

          const int pdg = SamplePdg();
          const bool isHadronic = (pdg == 2112);

          const double edepMeV = SampleEcalNoiseEnergyMeV(fRng);

          TVector3 pos = GetCrystalPosition(seqId);
          COMET::IGeometryId geomId;
          if (crystalId < 0 || !std::isfinite(pos.X()) || !std::isfinite(pos.Y())
              || !std::isfinite(pos.Z())
              || !ecalInfo.GetGeomIdFromGlobalPosition(pos, geomId)
              || !COMET::GeomId::ECAL::IsCrystal(geomId)
              || ecalInfo.GetSequenceIdFromGlobalPosition(pos) != seqId) {
            ++fNInvalidEcalGeometry;
            continue;
          }

          COMET::IG4HitCalo* hit = new COMET::IG4HitCalo();

          hit->SetRandomNoiseHit(
              -1,
              crystalId,
              pdg,
              pos,
              tNs * unit::ns,
              edepMeV * unit::MeV,
              isHadronic);

          ecalHits.push_back(hit);
          totalNoise++;
        }
      }
    }

    return totalNoise;
  }

  int AddStrawNoiseHits(COMET::IG4HitContainer& strawHits)
  {
    if (fStrawCenterRateMHzPerChannel == 0.0 && fStrawFloorRateMHzPerChannel == 0.0) return 0;
    const COMET::IStrawTrkGeom& geom = COMET::IGeomInfo::Get().StrawTrk();
  
    const int nStraws = geom.GetNumberOfStraws();
    const int nManifolds = geom.GetNumberOfManifolds();
    const int nStations = geom.GetNumberOfStations();
  
    int nAdded = 0;
  
    for (int stationId = 0; stationId < nStations; ++stationId) {
      for (int manifoldId = 0; manifoldId < nManifolds; ++manifoldId){
        for (int strawId = 0; strawId < nStraws; ++strawId){
          int seqId = geom.GetSequenceIdFromStrawTrk(stationId, manifoldId, strawId);
          const double rateMHz = GetStrawNoiseRateMHz(seqId, geom);
  
         if (rateMHz <= 0.0) continue;
  
          const double lambda = rateMHz * fStrawReadoutWindowNs * 1.0e-3;
  
          const int nNoise = fRng.Poisson(lambda);
  
          for (int i = 0; i < nNoise; ++i) {
            COMET::IG4HitGas* hit = MakeStrawNoiseHit(seqId, geom);
            strawHits.push_back(hit);
            ++nAdded;
          }
        }
      }
    }
  
    return nAdded;
  }

  double GetStrawNoiseRateMHz(int seqId, const COMET::IStrawTrkGeom& geom) const
  {
    int seq = seqId;
    const int strawId = geom.GetStrawIdFromSequenceId(seq);
  
    const int wireInColumn = strawId % 120;
  
    const double centerWire = 60.0;
    const double du = static_cast<double>(wireInColumn) - centerWire;
  
    const double sigma = std::max(1.0, fStrawSigmaWire);
  
    const double gaus =
        std::exp(-0.5 * du * du / (sigma * sigma));
  
    return fStrawFloorRateMHzPerChannel
         + fStrawCenterRateMHzPerChannel * gaus;
  }

  int SamplePdg()
  {
    if (fRng.Rndm() < fGammaFraction) {
      return 22;    // gamma
    }
    return 2112;    // neutron
  }

  double SampleEcalNoiseEnergyMeV(TRandom3& rng)
  {
      const double meanMeV  = 2.43;
      const double sigmaMeV = 1.322;
  
      const double k =
          (meanMeV * meanMeV) / (sigmaMeV * sigmaMeV);
  
      const double theta =
          (sigmaMeV * sigmaMeV) / meanMeV;
  
      double e = SampleGamma(rng, k, theta);
  
      const double eCapMeV = 30.0;
      const double eLowMeV = 1.5;
  
      while ((e < eLowMeV || e > eCapMeV)) {
          e = SampleGamma(rng, k, theta);
      }
  
      return e;
  }

  TVector3 GetCrystalPosition(UInt_t seqId)
  {
    // For the first version, put noise at the crystal center.
    // This is usually enough for waveform / clustering noise studies.
    return COMET::IGeomInfo::Get().ECAL().GetCrystalPosition(
        static_cast<int>(seqId));
  }

  COMET::IG4HitGas* MakeStrawNoiseHit(int seqId, const COMET::IStrawTrkGeom& geom)
  {
    const TVector3 wireEnd0 = geom.GetWireEnd0(seqId);
    const TVector3 wireEnd1 = geom.GetWireEnd1(seqId);
  
    TVector3 wireDir = wireEnd1 - wireEnd0;
    const double wireLength = wireDir.Mag();
  
    if (wireLength > 0.0) {
      wireDir *= (1.0 / wireLength);
    }
    else {
      wireDir = geom.GetWireDirection(seqId);
    }
  
    // Random point along wire.
    const double s = fRng.Uniform(0.0, wireLength);
    const TVector3 pocaOnWire = wireEnd0 + s * wireDir;
  
    const double strawRadius = 0.5 * geom.GetStrawDiameter();
  
    // Uniform ionization point inside tube cross section:
    // P(r) dr ∝ r dr, so r = R * sqrt(u).
    const double doca = strawRadius * std::sqrt(fRng.Rndm());
  
    // Build two unit vectors perpendicular to wire direction.
    TVector3 ref(1.0, 0.0, 0.0);
    if (std::abs(wireDir.Dot(ref)) > 0.9) {
      ref.SetXYZ(0.0, 1.0, 0.0);
    }
  
    TVector3 e1 = wireDir.Cross(ref);
    if (e1.Mag() <= 0.0) {
      e1.SetXYZ(0.0, 0.0, 1.0);
    }
    e1 = e1.Unit();
  
    TVector3 e2 = wireDir.Cross(e1).Unit();
  
    const double phi = fRng.Uniform(0.0, 2.0 * M_PI);
    const TVector3 radialDir = std::cos(phi) * e1 + std::sin(phi) * e2;
  
    const TVector3 poca = pocaOnWire + doca * radialDir;
  
    // Track direction chosen perpendicular to both wire and radial direction,
    // so that POCA/POCAOnWire are geometrically consistent.
    TVector3 trackDir = wireDir.Cross(radialDir);
    if (trackDir.Mag() <= 0.0) {
      trackDir = e1;
    }
    trackDir = trackDir.Unit();
  
    // Randomize sign.
    if (fRng.Rndm() < 0.5) {
      trackDir *= -1.0;
    }
  
    const double chordLength = 2.0 * std::sqrt(std::max(0.0, strawRadius * strawRadius - doca * doca));
  
    const TVector3 firstPos = poca - 0.5 * chordLength * trackDir;
  
    const TVector3 lastPos = poca + 0.5 * chordLength * trackDir;
  
    const double tNs = fRng.Uniform(0.0, fStrawReadoutWindowNs);
  
    const double time = tNs * unit::ns;
  
    // For electronics response, precise momentum is usually not important.
    const TVector3 mom = fStrawElectronMomentumMeV * unit::MeV * trackDir;
  
    const double edep = SampleStrawNoiseEnergyMeV() * unit::MeV;
  
    COMET::IG4HitGas* hit = new COMET::IG4HitGas();
  
    hit->SetRandomNoiseHit(
        -1,
        11,                 // electron
        poca,
        pocaOnWire,
        time,
        mom,
        firstPos,
        time,
        mom,
        lastPos,
        time,
        mom,
        chordLength,
        edep);
  
    return hit;
  }

  double SampleStrawNoiseEnergyMeV()
  {
    double e = fRng.Landau(fStrawEdepMpvMeV, fStrawEdepSigmaMeV);
  
    int tries = 0;
    while (e <= 0.00001 && tries < 100) {
      e = fRng.Landau(fStrawEdepMpvMeV, fStrawEdepSigmaMeV);
      ++tries;
    }
  
    if (e <= 0.0) {
      e = fStrawEdepMpvMeV;
    }
  
    return std::min(e, fStrawEdepCapMeV);
  }

private:
  double fStrawCenterRateMHzPerChannel;
  double fStrawFloorRateMHzPerChannel;
  double fStrawSigmaWire;
  double fStrawReadoutWindowNs;
  
  double fStrawElectronMomentumMeV;
  double fStrawEdepMpvMeV;
  double fStrawEdepSigmaMeV;
  double fStrawEdepCapMeV;
  
  long long fNAddedStrawTotal;

  double fReadoutWindowNs;
  double fEcalOccupancy;

  double fGammaFraction;

  UInt_t fSeed;
  bool   fVerbose;

  TRandom3 fRng;

  long long fNAddedTotal;
  long long fNInvalidEcalGeometry;

private:
  double SampleGamma(TRandom3& rng, double alpha, double theta)
  {
    // Gamma distribution with shape alpha and scale theta.
    // Mean = alpha * theta
    // Variance = alpha * theta^2
  
    if (alpha <= 0.0 || theta <= 0.0) {
      return 0.0;
    }
  
    // For alpha < 1, use:
    // Gamma(alpha) = Gamma(alpha + 1) * U^(1 / alpha)
    if (alpha < 1.0) {
      const double u = std::max(rng.Rndm(), 1.0e-12);
      return SampleGamma(rng, alpha + 1.0, theta) * std::pow(u, 1.0 / alpha);
    }
  
    // Marsaglia-Tsang method for alpha >= 1
    const double d = alpha - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
  
    while (true) {
      const double x = rng.Gaus(0.0, 1.0);
      const double v0 = 1.0 + c * x;
  
      if (v0 <= 0.0) {
        continue;
      }
  
      const double v = v0 * v0 * v0;
      const double u = rng.Rndm();
  
      if (u < 1.0 - 0.0331 * x * x * x * x) {
        return theta * d * v;
      }
  
      if (std::log(std::max(u, 1.0e-12))
          < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
        return theta * d * v;
      }
    }
  }
};

int main(int argc, char **argv)
{
  IUserLoop userLoop;
  cometEventLoop(argc, argv, userLoop);
}
