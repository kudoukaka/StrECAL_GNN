#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <limits>

#include <TString.h>
#include <TVector3.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <TSpectrum.h>
#include <TBox.h>

#include <ICOMETEvent.hxx>
#include <IHandle.hxx>
#include <HEPUnits.hxx>

#include <cometEventLoop.hxx>

#include "IDigitContainer.hxx"
#include "IWaveformDigit.hxx"
#include "IDetRespTime.hxx"
#include "IECALmcDigit.hxx"
#include "IStrawTrkmcDigit.hxx"
#include "IGeomInfo.hxx"
#include "IG4HitCalo.hxx"
#include "IG4HitGas.hxx"

/// ECAL-only detector-response analysis.
///
/// This version removes all straw / XT-relation diagnostics and only checks
/// the StrECAL ECAL hit map after detector response.  It reads digits/ECAL,
/// counts waveform peaks above baseline for each crystal, and writes both
/// accumulated maps and a configurable number of per-event maps.
class IUserLoop: public COMET::ICOMETEventLoopFunction
{
public:
  IUserLoop() :
    fECALDigitPath("digits/ECAL"),
    fStrawDigitPath("digits/StrawTrk"),
    fADCToVoltage(0.24414),
    fTickMultiplicity(25),
    fECALPeakThreshold(5.0),
    fECALPeakPolarity(+1),
    fECALBaselineNTicks(50),
    fECALMinPeakSeparationTicks(50),
    fStrawPulseThreshold(10.0),
    fStrawBaselineNTicks(50),
    fStrawMinPulseSeparationTicks(1),
    fNStrawColumns(0),
    fNStrawsPerColumn(0),
    fStrawGeometryReady(false),
    fECALTSpectrumSigma(2.0),
    fECALTSpectrumThresholdFraction(0.01),
    fMapBins(60),
    fMapHalfSizeMm(614.64),
    fECALZOffsetMm(7650.0),
    fNCrystals(512),
    fDrawECALOutline(true),
    fECALCrystalHalfSizeMm(0.0),
    fECALOutlineLineColor(15),
    fECALOutlineLineWidth(1),
    fMaxEventMaps(20),
    fVerbose(false),
    hECALDigitMap(nullptr),
    hECALPeakCountMap(nullptr),
    hECALActiveCrystalMap(nullptr),
    hECALDigitCrystalId(nullptr),
    hECALPeakCrystalId(nullptr),
    hECALDigitsPerEvent(nullptr),
    hECALPeaksPerEvent(nullptr),
    hECALActiveCrystalsPerEvent(nullptr),
    hECALPeakMultiplicityPerDigit(nullptr),
    hECALPeakTime(nullptr),
    hECALPeakAmplitude(nullptr),
    hStrawActiveColumnsPerEvent(nullptr),
    fNEvents(0),
    fNEventsWithoutECAL(0),
    fNECALDigits(0),
    fNECALDigitsWithPeaks(0),
    fNECALPeaks(0),
    fNEventsWithoutStraw(0),
    fNStrawDigits(0),
    fNStrawDigitsWithPulses(0),
    fNStrawPulses(0),
    fNEventMapsWritten(0)
  { }

  virtual ~IUserLoop() {}

  void Usage(void)
  {
    std::cout
      << "ECAL hit-map analysis for checking random-noise overlay.\n"
      << "\n"
      << "Options:\n"
      << "  -O ADCToVoltage=0.24414            conversion from ADC count to waveform unit\n"
      << "  -O ECALPeakThreshold=10.0          peak threshold above baseline in waveform unit\n"
      << "  -O ECALPeakPolarity=1              +1 positive, -1 negative, 0 abs(delta)\n"
      << "  -O ECALBaselineNTicks=50           leading ticks used for baseline\n"
      << "  -O ECALMinPeakSeparationTicks=10   minimum separation between peaks [ticks]\n"
      << "  -O ECALTSpectrumSigma=2.0          TSpectrum peak-search sigma [bins]\n"
      << "  -O ECALTSpectrumThresholdFraction=0.01 TSpectrum relative threshold wrt highest peak\n"
      << "  -O MapBins=60                      ECAL hit-map bins per axis\n"
      << "  -O MapHalfSizeMm=614.64            ECAL hit-map half range [mm]\n"
      << "  -O ECALZOffsetMm=7650.0            x-axis offset applied to crystal z position [mm]\n"
      << "  -O NCrystals=512                   number of crystals for crystalId histograms\n"
      << "  -O MaxEventMaps=20                 number of per-event ECAL maps to write\n"
      << "  -O StrawPulseThreshold=5.0         negative-pulse threshold above baseline\n"
      << "  -O StrawBaselineNTicks=50          leading ticks used for Straw baseline\n"
      << "  -O StrawMinPulseSeparationTicks=1  minimum separation between Straw pulses [ticks]\n"
      << "  -O Verbose=1                       print per-event counters\n";
  }

  virtual bool SetOption(std::string opt, std::string val)
  {
    if (opt == "ADCToVoltage") {
      fADCToVoltage = std::atof(val.c_str());
      return true;
    }

    if (opt == "ECALPeakThreshold") {
      fECALPeakThreshold = std::atof(val.c_str());
      if (fECALPeakThreshold < 0.0) fECALPeakThreshold = std::abs(fECALPeakThreshold);
      return true;
    }

    if (opt == "ECALPeakPolarity") {
      fECALPeakPolarity = std::atoi(val.c_str());
      if (fECALPeakPolarity > 0) fECALPeakPolarity = +1;
      else if (fECALPeakPolarity < 0) fECALPeakPolarity = -1;
      else fECALPeakPolarity = 0;
      return true;
    }

    if (opt == "ECALBaselineNTicks") {
      fECALBaselineNTicks = std::atoi(val.c_str());
      if (fECALBaselineNTicks <= 0) fECALBaselineNTicks = 1;
      return true;
    }

    if (opt == "ECALMinPeakSeparationTicks") {
      fECALMinPeakSeparationTicks = std::atoi(val.c_str());
      if (fECALMinPeakSeparationTicks <= 0) fECALMinPeakSeparationTicks = 1;
      return true;
    }

    if (opt == "StrawDigitPath") {
      fStrawDigitPath = val;
      return true;
    }

    if (opt == "StrawPulseThreshold") {
      fStrawPulseThreshold = std::atof(val.c_str());
      if (fStrawPulseThreshold < 0.0) {
        fStrawPulseThreshold = std::abs(fStrawPulseThreshold);
      }
      return true;
    }

    if (opt == "StrawBaselineNTicks") {
      fStrawBaselineNTicks = std::atoi(val.c_str());
      if (fStrawBaselineNTicks <= 0) fStrawBaselineNTicks = 1;
      return true;
    }

    if (opt == "StrawMinPulseSeparationTicks") {
      fStrawMinPulseSeparationTicks = std::atoi(val.c_str());
      if (fStrawMinPulseSeparationTicks <= 0) fStrawMinPulseSeparationTicks = 1;
      return true;
    }

    if (opt == "ECALTSpectrumSigma") {
      fECALTSpectrumSigma = std::atof(val.c_str());
      if (fECALTSpectrumSigma <= 0.0) fECALTSpectrumSigma = 2.0;
      return true;
    }

    if (opt == "ECALTSpectrumThresholdFraction") {
      fECALTSpectrumThresholdFraction = std::atof(val.c_str());
      if (fECALTSpectrumThresholdFraction < 0.0) fECALTSpectrumThresholdFraction = 0.0;
      if (fECALTSpectrumThresholdFraction > 1.0) fECALTSpectrumThresholdFraction = 1.0;
      return true;
    }

    if (opt == "MapBins") {
      fMapBins = std::atoi(val.c_str());
      if (fMapBins <= 0) fMapBins = 60;
      return true;
    }

    if (opt == "MapHalfSizeMm") {
      fMapHalfSizeMm = std::atof(val.c_str());
      if (fMapHalfSizeMm <= 0.0) fMapHalfSizeMm = 614.64;
      return true;
    }

    if (opt == "ECALZOffsetMm") {
      fECALZOffsetMm = std::atof(val.c_str());
      return true;
    }

    if (opt == "NCrystals") {
      fNCrystals = std::atoi(val.c_str());
      if (fNCrystals <= 0) fNCrystals = 512;
      return true;
    }

    if (opt == "MaxEventMaps") {
      fMaxEventMaps = std::atoi(val.c_str());
      if (fMaxEventMaps < 0) fMaxEventMaps = 0;
      return true;
    }

    if (opt == "Verbose") {
      fVerbose = (std::atoi(val.c_str()) != 0);
      return true;
    }

    return true;
  }

  bool operator () (COMET::ICOMETEvent& event)
  {
    ++fNEvents;

    COMET::IHandle<COMET::IDigitContainer> ecalDigits =
        event.Get<COMET::IDigitContainer>(fECALDigitPath.c_str());

    if (!ecalDigits) {
      ++fNEventsWithoutECAL;
      if (fVerbose) {
        std::cout << "[DetRespAna] no " << fECALDigitPath
                  << " in event " << event.GetEventId() << std::endl;
      }
    }
    else {
      int eventDigits = 0;
      int eventPeaks = 0;
      std::set<int> eventActiveSeqIds;
      std::map<int, int> eventPeakCountBySeqId;

      for (COMET::IDigitContainer::const_iterator it = ecalDigits->begin();
           it != ecalDigits->end(); ++it) {

        const COMET::IECALmcDigit* ecalDigit =
            dynamic_cast<const COMET::IECALmcDigit*>(*it);

        if (!ecalDigit) continue;

        double mapX = 0.0;
        double mapY = 0.0;
        int seqId = -1;
        if (!GetECALCrystalMapXY(*ecalDigit, seqId, mapX, mapY)) continue;

        ++eventDigits;
        ++fNECALDigits;

        hECALDigitMap->Fill(mapX, mapY);
        hECALDigitCrystalId->Fill(seqId);

        const std::vector<PeakInfo> peaks = FindECALWaveformPeaks(*ecalDigit);
        const int nPeaks = static_cast<int>(peaks.size());

        hECALPeakMultiplicityPerDigit->Fill(nPeaks);

        if (nPeaks <= 0) continue;

        eventActiveSeqIds.insert(seqId);
        ++fNECALDigitsWithPeaks;

        eventPeaks += nPeaks;
        fNECALPeaks += nPeaks;

        hECALPeakCountMap->Fill(mapX, mapY, static_cast<double>(nPeaks));
        hECALActiveCrystalMap->Fill(mapX, mapY);
        hECALPeakCrystalId->Fill(seqId, static_cast<double>(nPeaks));

        eventPeakCountBySeqId[seqId] += nPeaks;

        for (size_t i = 0; i < peaks.size(); ++i) {
          hECALPeakTime->Fill(peaks[i].timeNs);
          hECALPeakAmplitude->Fill(peaks[i].amplitude);
        }
      }

      hECALDigitsPerEvent->Fill(eventDigits);
      hECALPeaksPerEvent->Fill(eventPeaks);
      hECALActiveCrystalsPerEvent->Fill(static_cast<int>(eventActiveSeqIds.size()));

      if (fNEventMapsWritten < fMaxEventMaps && !eventPeakCountBySeqId.empty()) {
        WriteECALPeakMap(event.GetEventId(), eventPeakCountBySeqId);
        ++fNEventMapsWritten;
      }

      if (fVerbose) {
        std::cout << "[DetRespAna] event " << event.GetEventId()
                  << ": ECAL digits = " << eventDigits
                  << ", ECAL peaks = " << eventPeaks
                  << ", active crystals = " << eventActiveSeqIds.size()
                  << std::endl;
      }
    }

    AnalyzeStrawDigits(event);

    return true;
  }

  void Initialize(void)
  {
    const double mapMin = -fMapHalfSizeMm;
    const double mapMax =  fMapHalfSizeMm;

    hECALDigitMap = new TH2D(
        "hECALDigitMap",
        "Accumulated ECAL digit map;crystal X [mm];crystal Y [mm]",
        fMapBins, mapMin, mapMax,
        fMapBins, mapMin, mapMax);

    hECALPeakCountMap = new TH2D(
        "hECALPeakCountMap",
        "Accumulated ECAL waveform peak-count map;crystal X [mm];crystal Y [mm];peak count",
        fMapBins, mapMin, mapMax,
        fMapBins, mapMin, mapMax);

    hECALActiveCrystalMap = new TH2D(
        "hECALActiveCrystalMap",
        "Accumulated ECAL active-crystal map;crystal X [mm];crystal Y [mm];entries with #geq1 peak",
        fMapBins, mapMin, mapMax,
        fMapBins, mapMin, mapMax);

    hECALDigitCrystalId = new TH1D(
        "hECALDigitCrystalId",
        "ECAL digit crystal sequence ID;crystal sequence ID;digits",
        fNCrystals, 0.0, static_cast<double>(fNCrystals));

    hECALPeakCrystalId = new TH1D(
        "hECALPeakCrystalId",
        "ECAL waveform peak crystal sequence ID;crystal sequence ID;peaks",
        fNCrystals, 0.0, static_cast<double>(fNCrystals));

    hECALDigitsPerEvent = new TH1D(
        "hECALDigitsPerEvent",
        "ECAL digits per event;ECAL digits/event;events",
        200, 0.0, 200.0);

    hECALPeaksPerEvent = new TH1D(
        "hECALPeaksPerEvent",
        "ECAL waveform peaks per event;ECAL peaks/event;events",
        200, 0.0, 200.0);

    hECALActiveCrystalsPerEvent = new TH1D(
        "hECALActiveCrystalsPerEvent",
        "ECAL active crystals per event;crystals with #geq1 peak/event;events",
        200, 0.0, 200.0);

    hECALPeakMultiplicityPerDigit = new TH1D(
        "hECALPeakMultiplicityPerDigit",
        "ECAL waveform peak multiplicity per digit;peaks/digit;digits",
        20, 0.0, 20.0);

    hECALPeakTime = new TH1D(
        "hECALPeakTime",
        "ECAL waveform peak time;time [ns];peaks",
        1200, -100.0, 1100.0);

    hECALPeakAmplitude = new TH1D(
        "hECALPeakAmplitude",
        "ECAL waveform peak amplitude above baseline;amplitude;peaks",
        400, 0.0, 400.0);

  }

  void Finalize(COMET::ICOMETOutput*)
  {
    std::cout << "[DetRespAna] events processed          = " << fNEvents << std::endl;
    std::cout << "[DetRespAna] events without ECAL       = " << fNEventsWithoutECAL << std::endl;
    std::cout << "[DetRespAna] ECAL digits               = " << fNECALDigits << std::endl;
    std::cout << "[DetRespAna] ECAL digits with peaks    = " << fNECALDigitsWithPeaks << std::endl;
    std::cout << "[DetRespAna] ECAL waveform peaks       = " << fNECALPeaks << std::endl;

    if (fNEvents > 0) {
      std::cout << "[DetRespAna] mean ECAL digits/event    = "
                << static_cast<double>(fNECALDigits) / static_cast<double>(fNEvents)
                << std::endl;
      std::cout << "[DetRespAna] mean ECAL peaks/event     = "
                << static_cast<double>(fNECALPeaks) / static_cast<double>(fNEvents)
                << std::endl;
    }

    std::cout << "[DetRespAna] events without Straw      = " << fNEventsWithoutStraw << std::endl;
    std::cout << "[DetRespAna] Straw digits              = " << fNStrawDigits << std::endl;
    std::cout << "[DetRespAna] Straw digits with pulses  = " << fNStrawDigitsWithPulses << std::endl;
    std::cout << "[DetRespAna] Straw threshold pulses    = " << fNStrawPulses << std::endl;

    if (fNEvents > 0) {
      std::cout << "[DetRespAna] mean Straw digits/event   = "
                << static_cast<double>(fNStrawDigits) / static_cast<double>(fNEvents)
                << std::endl;
      std::cout << "[DetRespAna] mean Straw pulses/event   = "
                << static_cast<double>(fNStrawPulses) / static_cast<double>(fNEvents)
                << std::endl;
    }

    WriteCanvas("cECALDigitMap", hECALDigitMap, "COLZ");
    WriteCanvas("cECALPeakCountMap", hECALPeakCountMap, "COLZ");
    WriteCanvas("cECALActiveCrystalMap", hECALActiveCrystalMap, "COLZ");
    WriteCanvas("cECALDigitCrystalId", hECALDigitCrystalId, "");
    WriteCanvas("cECALPeakCrystalId", hECALPeakCrystalId, "");
    WriteCanvas("cECALDigitsPerEvent", hECALDigitsPerEvent, "");
    WriteCanvas("cECALPeaksPerEvent", hECALPeaksPerEvent, "");
    WriteCanvas("cECALActiveCrystalsPerEvent", hECALActiveCrystalsPerEvent, "");
    WriteCanvas("cECALPeakMultiplicityPerDigit", hECALPeakMultiplicityPerDigit, "");
    WriteCanvas("cECALPeakTime", hECALPeakTime, "");
    WriteCanvas("cECALPeakAmplitude", hECALPeakAmplitude, "");

    if (fStrawGeometryReady) {
      WriteCanvas("cStrawActiveColumnsPerEvent", hStrawActiveColumnsPerEvent, "");
  
      for (int col = 0; col < fNStrawColumns; ++col) {
        TString cname;
  
        cname.Form("cStrawOccupiedChannelsPerEvent_col%02d", col);
        WriteCanvas(cname, hStrawOccupiedChannelsPerEvent[col], "");
  
        cname.Form("cStrawPulsesPerEvent_col%02d", col);
        WriteCanvas(cname, hStrawPulsesPerEvent[col], "");
  
        cname.Form("cStrawOccupiedWireId_col%02d", col);
        WriteCanvas(cname, hStrawOccupiedWireId[col], "");
  
        cname.Form("cStrawPulseWireId_col%02d", col);
        WriteCanvas(cname, hStrawPulseWireId[col], "");
  
        cname.Form("cStrawPulseTime_col%02d", col);
        WriteCanvas(cname, hStrawPulseTime[col], "");
  
        cname.Form("cStrawPulseAmplitude_col%02d", col);
        WriteCanvas(cname, hStrawPulseAmplitude[col], "");

        cname.Form("cStrawOccupiedChannelsPerEventCentral_col%02d", col);
        WriteCanvas(cname, hStrawOccupiedChannelsPerEventCentral[col], "");
      }
    } else {
      std::cout << "[DetRespAna] Straw geometry was not initialized; "
                << "no Straw occupancy canvases written."
                << std::endl;
    }
  }

private:
  struct PeakInfo {
    int sampleIndex;
    double timeNs;
    double amplitude;
  };

  struct StrawChannelInfo {
    int columnId;
    int stationId;
    int manifoldId;
    int strawId;
  };

  struct StrawPulseInfo {
    int sampleIndex;
    double timeNs;
    double amplitude;
  };

  std::vector<PeakInfo> FindECALWaveformPeaks(const COMET::IECALmcDigit& digit) const
  {
    std::vector<PeakInfo> peaks;

    const COMET::IWaveformDigit::TWfADC& adcs = digit.GetADCs();
    const size_t nSamples = adcs.size();

    if (nSamples < 3) return peaks;

    const size_t nBase =
        std::min(nSamples, static_cast<size_t>(fECALBaselineNTicks));

    if (nBase == 0) return peaks;

    double baseline = 0.0;
    for (size_t i = 0; i < nBase; ++i) {
      baseline += static_cast<double>(adcs[i] * fADCToVoltage);
    }
    baseline /= static_cast<double>(nBase);

    TString hname;
    hname.Form("hTmpECALWaveform_%p", static_cast<const void*>(&digit));

    TH1D hWave(
        hname,
        "temporary ECAL waveform for TSpectrum",
        static_cast<int>(nSamples),
        -0.5,
        static_cast<double>(nSamples) - 0.5);

    hWave.SetDirectory(nullptr);
    hWave.SetStats(0);

    double maxAmplitude = 0.0;

    for (size_t i = 0; i < nSamples; ++i) {
      const double raw =
          static_cast<double>(adcs[i] * fADCToVoltage) - baseline;

      double y = 0.0;

      if (fECALPeakPolarity > 0) {
        y = raw;
      }
      else if (fECALPeakPolarity < 0) {
        y = -raw;
      }
      else {
        y = std::abs(raw);
      }

      // TSpectrum searches positive peaks.
      // Negative contents are not useful for this peak finder.
      if (y < 0.0) y = 0.0;

      hWave.SetBinContent(static_cast<int>(i) + 1, y);

      if (y > maxAmplitude) {
        maxAmplitude = y;
      }
    }

    // Absolute threshold used by this analysis.
    // TSpectrum's own threshold is relative to the highest peak, so keep both.
    if (maxAmplitude < fECALPeakThreshold) {
      return peaks;
    }

    const int nMaxPeaks =
        std::max(10, std::min(1000, static_cast<int>(nSamples / 2)));

    TSpectrum spectrum(nMaxPeaks);

    const Int_t nFound = spectrum.Search(
        &hWave,
        fECALTSpectrumSigma,
        "nobackground nodraw goff",
        fECALTSpectrumThresholdFraction);

    if (nFound <= 0) {
      return peaks;
    }

    Double_t* posX = spectrum.GetPositionX();

    std::vector<PeakInfo> candidates;

    for (Int_t ip = 0; ip < nFound; ++ip) {
      const int sampleIndex =
          static_cast<int>(std::floor(posX[ip] + 0.5));

      if (sampleIndex < 0 ||
          sampleIndex >= static_cast<int>(nSamples)) {
        continue;
      }

      const double amplitude =
          hWave.GetBinContent(sampleIndex + 1);

      if (amplitude < fECALPeakThreshold) {
        continue;
      }

      const int tick =
          digit.GetFirstTimeTick() + sampleIndex;

      PeakInfo peak;
      peak.sampleIndex = sampleIndex;
      peak.timeNs =
          IDetRespTime::Get().GetTimeOfTickCentre(tick, fTickMultiplicity)
          / unit::ns;
      peak.amplitude = amplitude;

      candidates.push_back(peak);
    }

    if (candidates.empty()) {
      return peaks;
    }

    // First select strongest peaks, enforcing minimum separation.
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const PeakInfo& a, const PeakInfo& b) {
          return a.amplitude > b.amplitude;
        });

    for (size_t i = 0; i < candidates.size(); ++i) {
      bool tooClose = false;

      for (size_t j = 0; j < peaks.size(); ++j) {
        if (std::abs(candidates[i].sampleIndex - peaks[j].sampleIndex)
            < fECALMinPeakSeparationTicks) {
          tooClose = true;
          break;
        }
      }

      if (!tooClose) {
        peaks.push_back(candidates[i]);
      }
    }

    // Return peaks in time order.
    std::sort(
        peaks.begin(),
        peaks.end(),
        [](const PeakInfo& a, const PeakInfo& b) {
          return a.sampleIndex < b.sampleIndex;
        });

    return peaks;
  }

  bool GetECALCrystalMapXY(const COMET::IECALmcDigit& digit,
                           int& seqId,
                           double& x,
                           double& y) const
  {
    const auto& ecalInfo = COMET::IGeomInfo::ECAL();

    seqId = ecalInfo.GetSequenceIdFromChannelId(digit.GetChannelId());

    if (seqId < 0) return false;

    TVector3 pos = COMET::IGeomInfo::Get().ECAL().GetCrystalPosition(seqId);

    x = pos.Z() / unit::mm - fECALZOffsetMm;
    y = pos.Y() / unit::mm;

    return true;
  }

  bool GetECALCrystalMapXYFromSeqId(int seqId, double& x, double& y) const
  {
    if (seqId < 0) return false;

    TVector3 pos = COMET::IGeomInfo::Get().ECAL().GetCrystalPosition(seqId);

    x = pos.Z() / unit::mm - fECALZOffsetMm;
    y = pos.Y() / unit::mm;

    return true;
  }

  void WriteECALPeakMap(int eventId,
                        const std::map<int, int>& eventPeakCountBySeqId) const
  {
    const double mapMin = -fMapHalfSizeMm;
    const double mapMax =  fMapHalfSizeMm;

    TString hname;
    hname.Form("hECALPeakMap_evt%d_%03lld", eventId, fNEventMapsWritten);

    TString htitle;
    htitle.Form("ECAL waveform peak-count map event %d;crystal X [mm];crystal Y [mm];peak count",
                eventId);

    TH2D* hMap = new TH2D(hname, htitle,
                          fMapBins, mapMin, mapMax,
                          fMapBins, mapMin, mapMax);

    for (std::map<int, int>::const_iterator it = eventPeakCountBySeqId.begin();
         it != eventPeakCountBySeqId.end(); ++it) {

      double x = 0.0;
      double y = 0.0;
      if (!GetECALCrystalMapXYFromSeqId(it->first, x, y)) continue;

      hMap->Fill(x, y, static_cast<double>(it->second));
    }

    hMap->SetStats(0);
    hMap->Write();

    TString cname;
    cname.Form("cECALPeakMap_evt%d_%03lld", eventId, fNEventMapsWritten);

    TCanvas* c = new TCanvas(cname, cname, 1000, 850);
    c->cd();
    hMap->SetMinimum(0.0);
    hMap->Draw("COLZ TEXT");
    DrawECALCrystalOutline();
    c->Write();
  }

    double GetMinPositiveSpacing(std::vector<double> values) const
  {
    if (values.size() < 2) return -1.0;

    std::sort(values.begin(), values.end());

    double minSpacing = std::numeric_limits<double>::max();

    for (size_t i = 1; i < values.size(); ++i) {
      const double d = std::abs(values[i] - values[i - 1]);

      // Ignore numerical duplicates.
      if (d > 1.0e-3 && d < minSpacing) {
        minSpacing = d;
      }
    }

    if (minSpacing == std::numeric_limits<double>::max()) {
      return -1.0;
    }

    return minSpacing;
  }

  void GetECALCrystalBoxHalfSize(double& halfX, double& halfY) const
  {
    if (fECALCrystalHalfSizeMm > 0.0) {
      halfX = fECALCrystalHalfSizeMm;
      halfY = fECALCrystalHalfSizeMm;
      return;
    }

    const auto& ecalInfo = COMET::IGeomInfo::ECAL();

    std::vector<double> xs;
    std::vector<double> ys;

    xs.reserve(fNCrystals);
    ys.reserve(fNCrystals);

    for (int block = 0; block < fNCrystals / 4; ++block){
      for (int crystal = 0; crystal < 4; ++crystal) {
        int seqId = ecalInfo.GetSequenceIdFromECAL(block, crystal);
        double x = 0.0;
        double y = 0.0;

        if (!GetECALCrystalMapXYFromSeqId(seqId, x, y)) continue;
  
        xs.push_back(x);
        ys.push_back(y);
      }
    }

    const double dx = GetMinPositiveSpacing(xs);
    const double dy = GetMinPositiveSpacing(ys);

    // Use 90% of nearest-neighbour spacing as full box size.
    // This leaves a small visual gap between crystals.
    halfX = (dx > 0.0) ? 0.45 * dx : 10.0;
    halfY = (dy > 0.0) ? 0.45 * dy : halfX;
  }

  void DrawECALCrystalOutline() const
  {
    if (!fDrawECALOutline) return;

    double halfX = 0.0;
    double halfY = 0.0;
    GetECALCrystalBoxHalfSize(halfX, halfY);

    for (int seqId = 0; seqId < fNCrystals; ++seqId) {
      double x = 0.0;
      double y = 0.0;

      if (!GetECALCrystalMapXYFromSeqId(seqId, x, y)) continue;

      TBox* box = new TBox(
          x - halfX,
          y - halfY,
          x + halfX,
          y + halfY);

      box->SetFillStyle(0);
      box->SetLineColor(fECALOutlineLineColor);
      box->SetLineWidth(fECALOutlineLineWidth);
      box->SetBit(TObject::kCanDelete);

      box->Draw("same");
    }
  }

  void BuildStrawChannelMap()
  {
    fStrawChannelInfo.clear();

    const auto& geom = COMET::IGeomInfo::StrawTrk();

    const int nStraws = geom.GetNumberOfStraws();
    const int nManifolds = geom.GetNumberOfManifolds();
    const int nStations = geom.GetNumberOfStations();

    fNStrawColumns = nStations * nManifolds;
    fNStrawsPerColumn = nStraws;

    for (int stationId = 0; stationId < nStations; ++stationId) {
      for (int manifoldId = 0; manifoldId < nManifolds; ++manifoldId) {
        for (int strawId = 0; strawId < nStraws; ++strawId) {
          const int columnId = stationId * nManifolds + manifoldId;
          int st = stationId;
          int mf = manifoldId;
          int sw = strawId;

          const int seqId = geom.GetSequenceIdFromStrawTrk(st, mf, sw);

          StrawChannelInfo info;
          info.columnId = columnId;
          info.stationId = stationId;
          info.manifoldId = manifoldId;
          info.strawId = strawId;

          fStrawChannelInfo[seqId] = info;
        }
      }
    }

    std::cout << "[DetRespAna] Straw geometry map built: "
              << "stations = " << nStations
              << ", manifolds = " << nManifolds
              << ", straws/column = " << nStraws
              << ", columns = " << fNStrawColumns
              << std::endl;
  }

  bool IsCentralStrawId(int strawId) const
  {
    return ((strawId >= 40 && strawId <= 80) ||
            (strawId >= 160 && strawId <= 200));
  }

  bool GetStrawColumnInfoFromDigit(const COMET::IWaveformDigit& digit,
                                   StrawChannelInfo& info) const
  {
    const auto& geom = COMET::IGeomInfo::StrawTrk();

    int seqId = geom.GetSequenceIdFromChannelId(digit.GetChannelId());

    std::map<int, StrawChannelInfo>::const_iterator it =
        fStrawChannelInfo.find(seqId);

    if (it == fStrawChannelInfo.end()) {
      return false;
    }

    info = it->second;
    return true;
  }

  std::vector<StrawPulseInfo> FindStrawNegativePulses(
      const COMET::IWaveformDigit& digit) const
  {
    std::vector<StrawPulseInfo> pulses;

    const COMET::IWaveformDigit::TWfADC& adcs = digit.GetADCs();
    const size_t nSamples = adcs.size();

    if (nSamples == 0) return pulses;

    const size_t nBase =
        std::min(nSamples, static_cast<size_t>(fStrawBaselineNTicks));

    if (nBase == 0) return pulses;

    double baseline = 0.0;
    for (size_t i = 0; i < nBase; ++i) {
      baseline += static_cast<double>(adcs[i] * fADCToVoltage);
    }
    baseline /= static_cast<double>(nBase);

    int lastPulseIndex = -999999;

    size_t i = 0;
    while (i < nSamples) {
      const double y =
          static_cast<double>(adcs[i] * fADCToVoltage) - baseline;

      const double amp = -y;  // Straw response is a negative 1 ns pulse.

      if (amp < fStrawPulseThreshold) {
        ++i;
        continue;
      }

      size_t bestIndex = i;
      double bestAmp = amp;

      const size_t start = i;

      while (i < nSamples) {
        const double yy =
            static_cast<double>(adcs[i] * fADCToVoltage) - baseline;
        const double aa = -yy;

        if (aa < fStrawPulseThreshold) break;

        if (aa > bestAmp) {
          bestAmp = aa;
          bestIndex = i;
        }

        ++i;
      }

      if (static_cast<int>(bestIndex) - lastPulseIndex
          < fStrawMinPulseSeparationTicks) {
        continue;
      }

      const int tick = digit.GetFirstTimeTick() + static_cast<int>(bestIndex);

      StrawPulseInfo pulse;
      pulse.sampleIndex = static_cast<int>(bestIndex);
      pulse.timeNs =
          IDetRespTime::Get().GetTimeOfTickCentre(tick, fTickMultiplicity)
          / unit::ns;
      pulse.amplitude = bestAmp;

      pulses.push_back(pulse);
      lastPulseIndex = static_cast<int>(bestIndex);

      if (i == start) ++i;
    }

    return pulses;
  }

  void AnalyzeStrawDigits(COMET::ICOMETEvent& event)
  {
    COMET::IHandle<COMET::IDigitContainer> strawDigits =
        event.Get<COMET::IDigitContainer>(fStrawDigitPath.c_str());

    if (!strawDigits) {
      ++fNEventsWithoutStraw;

      if (fVerbose) {
        std::cout << "[DetRespAna] no " << fStrawDigitPath
                  << " in event " << event.GetEventId() << std::endl;
      }

      return;
    }

    if (!EnsureStrawGeometryAndHistograms()) {
      ++fNEventsWithoutStraw;
      return;
    }

    std::vector<std::set<int> > occupiedStrawsByColumn(fNStrawColumns);
    std::vector<std::set<int> > occupiedCentralStraws(fNStrawColumns);
    std::vector<int> pulsesByColumn(fNStrawColumns, 0);

    for (COMET::IDigitContainer::const_iterator it = strawDigits->begin();
         it != strawDigits->end(); ++it) {

      const COMET::IWaveformDigit* strawDigit =
          dynamic_cast<const COMET::IWaveformDigit*>(*it);

      if (!strawDigit) continue;

      StrawChannelInfo info;
      if (!GetStrawColumnInfoFromDigit(*strawDigit, info)) continue;

      if (info.columnId < 0 || info.columnId >= fNStrawColumns) continue;

      ++fNStrawDigits;

      const std::vector<StrawPulseInfo> pulses =
          FindStrawNegativePulses(*strawDigit);

      const int nPulses = static_cast<int>(pulses.size());

      if (nPulses <= 0) continue;

      ++fNStrawDigitsWithPulses;
      fNStrawPulses += nPulses;

      occupiedStrawsByColumn[info.columnId].insert(info.strawId);
      hStrawOccupiedWireId[info.columnId]->Fill(info.strawId);
      if (IsCentralStrawId(info.strawId)) {
        occupiedCentralStraws[info.columnId].insert(info.strawId);
      }

      pulsesByColumn[info.columnId] += nPulses;

      hStrawPulseWireId[info.columnId]->Fill(
          info.strawId,
          static_cast<double>(nPulses));

      for (size_t ip = 0; ip < pulses.size(); ++ip) {
        hStrawPulseTime[info.columnId]->Fill(pulses[ip].timeNs);
        hStrawPulseAmplitude[info.columnId]->Fill(pulses[ip].amplitude);
      }
    }

    int activeColumns = 0;

    for (int col = 0; col < fNStrawColumns; ++col) {
      const int nOccupied = static_cast<int>(occupiedStrawsByColumn[col].size());
      const int nOccupiedCenter = static_cast<int>(occupiedCentralStraws[col].size());

      hStrawOccupiedChannelsPerEvent[col]->Fill(nOccupied);
      hStrawOccupiedChannelsPerEventCentral[col]->Fill(nOccupiedCenter);
      hStrawPulsesPerEvent[col]->Fill(pulsesByColumn[col]);

      if (nOccupied > 0) ++activeColumns;
    }

    hStrawActiveColumnsPerEvent->Fill(activeColumns);

    if (fVerbose) {
      int totalOccupied = 0;
      int totalPulses = 0;

      for (int col = 0; col < fNStrawColumns; ++col) {
        totalOccupied += static_cast<int>(occupiedStrawsByColumn[col].size());
        totalPulses += pulsesByColumn[col];
      }

      std::cout << "[DetRespAna] event " << event.GetEventId()
                << ": Straw occupied channels = " << totalOccupied
                << ", Straw pulses = " << totalPulses
                << ", active columns = " << activeColumns
                << std::endl;
    }
  }

  bool EnsureStrawGeometryAndHistograms()
  {
    if (fStrawGeometryReady) return true;
  
    try {
      BuildStrawChannelMap();
    }
    catch (const std::exception& e) {
      std::cerr << "[DetRespAna] failed to build Straw geometry map: "
                << e.what() << std::endl;
      return false;
    }
  
    if (fNStrawColumns <= 0 || fNStrawsPerColumn <= 0) {
      std::cerr << "[DetRespAna] invalid Straw geometry: "
                << "columns = " << fNStrawColumns
                << ", straws/column = " << fNStrawsPerColumn
                << std::endl;
      return false;
    }
  
    hStrawActiveColumnsPerEvent = new TH1D(
        "hStrawActiveColumnsPerEvent",
        "Straw active columns per event;columns with #geq1 occupied channel/event;events",
        fNStrawColumns + 1,
        -0.5,
        static_cast<double>(fNStrawColumns) + 0.5);
  
    hStrawOccupiedChannelsPerEvent.resize(fNStrawColumns, nullptr);
    hStrawPulsesPerEvent.resize(fNStrawColumns, nullptr);
    hStrawOccupiedWireId.resize(fNStrawColumns, nullptr);
    hStrawPulseWireId.resize(fNStrawColumns, nullptr);
    hStrawPulseTime.resize(fNStrawColumns, nullptr);
    hStrawPulseAmplitude.resize(fNStrawColumns, nullptr);
    hStrawOccupiedChannelsPerEventCentral.resize(fNStrawColumns, nullptr);
  
    for (int col = 0; col < fNStrawColumns; ++col) {
      TString name;
      TString title;
  
      name.Form("hStrawOccupiedChannelsPerEvent_layer%02d", col);
      title.Form("Straw occupied channels per event, layer %02d;occupied channels/event;events", col);
      hStrawOccupiedChannelsPerEvent[col] = new TH1D(
          name,
          title,
          fNStrawsPerColumn + 1,
          -0.5,
          static_cast<double>(fNStrawsPerColumn) + 0.5);
  
      name.Form("hStrawPulsesPerEvent_layer%02d", col);
      title.Form("Straw threshold pulses per event, layer %02d;pulses/event;events", col);
      hStrawPulsesPerEvent[col] = new TH1D(
          name,
          title,
          200,
          0.0,
          200.0);
  
      name.Form("hStrawOccupiedWireId_layer%02d", col);
      title.Form("Straw occupied wireId, layer %02d;wireId in layer;occupied channel entries", col);
      hStrawOccupiedWireId[col] = new TH1D(
          name,
          title,
          fNStrawsPerColumn,
          0.0,
          static_cast<double>(fNStrawsPerColumn));
  
      name.Form("hStrawPulseWireId_layer%02d", col);
      title.Form("Straw pulse wireId, layer %02d;wireId in layer;pulses", col);
      hStrawPulseWireId[col] = new TH1D(
          name,
          title,
          fNStrawsPerColumn,
          0.0,
          static_cast<double>(fNStrawsPerColumn));
  
      name.Form("hStrawPulseTime_layer%02d", col);
      title.Form("Straw pulse time, layer %02d;time [ns];pulses", col);
      hStrawPulseTime[col] = new TH1D(
          name,
          title,
          1200,
          -100.0,
          1100.0);
  
      name.Form("hStrawPulseAmplitude_layer%02d", col);
      title.Form("Straw negative-pulse amplitude, layer %02d;baseline - sample;pulses", col);
      hStrawPulseAmplitude[col] = new TH1D(
          name,
          title,
          400,
          0.0,
          400.0);

      name.Form("hStrawOccupiedChannelsPerEventCentral_layer%02d", col);
      title.Form("Straw occupied cnetral channels per event, layer %02d;occupied channels/event;events", col);
      hStrawOccupiedChannelsPerEventCentral[col] = new TH1D(
          name,
          title,
          fNStrawsPerColumn + 1,
          -0.5,
          static_cast<double>(fNStrawsPerColumn)/3 + 0.5);
    }
  
    fStrawGeometryReady = true;
  
    std::cout << "[DetRespAna] Straw histograms initialized: "
              << " layers = " << fNStrawColumns
              << ", straws/layer = " << fNStrawsPerColumn
              << std::endl;
  
    return true;
  }

  void WriteECALMapCanvas(const char* name, TH2D* hist, const char* drawOpt) const
  {
    if (!hist) return;

    TCanvas* c = new TCanvas(name, name, 1000, 850);
    c->cd();

    hist->SetStats(0);
    hist->SetMinimum(0.0);
    hist->Draw(drawOpt);

    DrawECALCrystalOutline();

    c->Write();
  }

  void WriteCanvas(const TString& name, TObject* obj, const char* drawOpt) const
  {
    if (!obj) return;

    TCanvas* c = new TCanvas(name, name, 900, 700);
    c->cd();
    obj->Draw(drawOpt);
    c->Write();
  }

private:
  std::string fECALDigitPath;
  std::string fStrawDigitPath;

  double fADCToVoltage;
  int    fTickMultiplicity;

  double fECALPeakThreshold;
  int    fECALPeakPolarity;
  int    fECALBaselineNTicks;
  int    fECALMinPeakSeparationTicks;

  double fStrawPulseThreshold;
  int    fStrawBaselineNTicks;
  int    fStrawMinPulseSeparationTicks;

  int    fNStrawColumns;
  int    fNStrawsPerColumn;

  std::map<int, StrawChannelInfo> fStrawChannelInfo;
  bool fStrawGeometryReady;

  double fECALTSpectrumSigma;
  double fECALTSpectrumThresholdFraction;

  int    fMapBins;
  double fMapHalfSizeMm;
  double fECALZOffsetMm;
  int    fNCrystals;
  bool   fDrawECALOutline;
  double fECALCrystalHalfSizeMm;
  int    fECALOutlineLineColor;
  int    fECALOutlineLineWidth;

  long long fMaxEventMaps;
  bool      fVerbose;

  TH2D* hECALDigitMap;
  TH2D* hECALPeakCountMap;
  TH2D* hECALActiveCrystalMap;

  TH1D* hECALDigitCrystalId;
  TH1D* hECALPeakCrystalId;
  TH1D* hECALDigitsPerEvent;
  TH1D* hECALPeaksPerEvent;
  TH1D* hECALActiveCrystalsPerEvent;
  TH1D* hECALPeakMultiplicityPerDigit;
  TH1D* hECALPeakTime;
  TH1D* hECALPeakAmplitude;

  TH1D* hStrawActiveColumnsPerEvent;

  std::vector<TH1D*> hStrawOccupiedChannelsPerEvent;
  std::vector<TH1D*> hStrawPulsesPerEvent;
  std::vector<TH1D*> hStrawOccupiedWireId;
  std::vector<TH1D*> hStrawPulseWireId;
  std::vector<TH1D*> hStrawPulseTime;
  std::vector<TH1D*> hStrawPulseAmplitude;
  std::vector<TH1D*> hStrawOccupiedChannelsPerEventCentral;

  long long fNEvents;
  long long fNEventsWithoutECAL;
  long long fNECALDigits;
  long long fNECALDigitsWithPeaks;
  long long fNECALPeaks;
  long long fNEventMapsWritten;

  long long fNEventsWithoutStraw;
  long long fNStrawDigits;
  long long fNStrawDigitsWithPulses;
  long long fNStrawPulses;
};

int main(int argc, char **argv)
{
  IUserLoop userLoop;
  cometEventLoop(argc, argv, userLoop);
}
