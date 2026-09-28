#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>
#include <iostream>
#include <iomanip>
#include <string>
void inspect(const char* species,const char* suffix,Long64_t entry){
 std::string path="/rhome/comet/davidz/ReconData/datasets/ecal_clustering/truthmaker/"+std::string(species)+"_20to120MeV_10cmPlateGun_0To10deg/oa_g4_"+species+"_00000001-0018_"+suffix+"_SG4_000_truthmaker.root";
 TFile f(path.c_str());std::cout<<"EXAMPLE "<<species<<" entry="<<entry<<std::endl;
 TTreeReader r((TTree*)f.Get("ecalTruthClusters"));TTreeReaderValue<Long64_t> event(r,"event");
 TTreeReaderArray<int> cid(r,"clusterId"),pid(r,"pid"),primary(r,"primaryId");
 TTreeReaderArray<double> e(r,"edepMeV"),t0(r,"rawT0Ns"),inc(r,"incidentTimeNs"),ke(r,"incidentKineticEnergyMeV"),mom(r,"incidentMom");
 r.SetEntry(entry);double earliest=1e99;
 for(size_t i=0;i<cid.GetSize();i++){earliest=std::min(earliest,inc[i]);std::cout<<"cluster event="<<*event<<" cid="<<cid[i]<<" primary="<<primary[i]<<" pid="<<pid[i]<<" edep="<<e[i]<<" t0="<<t0[i]<<" incident_t="<<inc[i]<<" incident_KE="<<ke[i]<<" incident_p="<<mom[i]<<std::endl;}
 TTreeReader d((TTree*)f.Get("ecalTruthDigits"));TTreeReaderArray<double> dt(d,"contributorTimeNs"),de(d,"contributorEdepMeV");d.SetEntry(entry);double prompt=0,late=0;for(size_t i=0;i<dt.GetSize();i++){if(dt[i]<=earliest+30)prompt+=de[i];else late+=de[i];}
 std::cout<<"digit contributors Edep prompt30ns="<<prompt<<" later="<<late<<" firstTruthIncident="<<earliest<<std::endl;
 TTreeReader p((TTree*)f.Get("ecalRecoPulses"));TTreeReaderArray<double> pt(p,"fitPeakTimeNs"),pa(p,"fitAmplitude"),mt(p,"matchedTruthEdepMeV");TTreeReaderArray<int> pc(p,"clusterId"),di(p,"digitIndex"),status(p,"fitStatus");p.SetEntry(entry);
 for(size_t i=0;i<pt.GetSize();i++)std::cout<<"pulse digit="<<di[i]<<" time="<<pt[i]<<" amp="<<pa[i]<<" truthE="<<mt[i]<<" cluster="<<pc[i]<<" status="<<status[i]<<std::endl;
}
void inspect_late_examples(){std::cout<<std::setprecision(10);inspect("mum","wvw6gifbde3g",36795);inspect("mum","wvw6gifbde3g",33742);inspect("pim","2fynbsuq7oil",35050);inspect("pim","2fynbsuq7oil",21672);}
