//
// created:       2016-10-07
// last changed:  2016-10-07
//
// based on AliTRDcheckConfig.C
//


#if !defined(__CINT__) || defined(__MAKECINT__)

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <TFile.h>
#include <TString.h>

#include <AliTRDCalDet.h>
#include <AliTRDSensorArray.h>
#include <TCanvas.h>
#include <AliGRPObject.h>
#include <AliCDBEntry.h>
#include <AliTRDCalDCS.h>
#include <AliTRDCalDCSv2.h>
#include <AliCDBManager.h>

#endif

using std::cout;
using std::endl;

struct runinfo_t
{

  // GRP info
  TTimeStamp start_time;
  TTimeStamp end_time;

  float  cavern_pressure;
  //float  surface_pressure;
  //int    detector_mask;

  float anode_voltage[540];
  float drift_voltage[540];

  float gain[540];
  float vdrift[540];
  float ExB[540];

};

//__________________________________________________________________________
void query(Int_t year=2016, Int_t run=265377)
{

    ofstream myfile;
    myfile.open("example.txt");

  // set up the connection to the OCDB
  AliCDBManager* man = AliCDBManager::Instance();

  if (0) {
    man->SetDefaultStorage
      (Form("local:///cvmfs/alice-ocdb.cern.ch/calibration/data/%d/OCDB/",year));
  } else {
    man->SetDefaultStorage(Form("alien://folder=/alice/data/%d/OCDB/",year));
  }

  man->SetCacheFlag(kTRUE);
  man->SetRun(run);

  runinfo_t runinfo;

  // -----------------------------------------------------------------------
  // Get the GRP data. Only runs with a corresponding GRP entry in the OCDB
  // will be processed
  AliCDBEntry *entry = man->Get("GRP/GRP/Data",run);
  //AliGRPObject* grp = (AliGRPObject*)entry->GetObject();
  AliGRPObject* grp = (AliGRPObject*)entry->GetObject();

  if (!grp) return;

  TTimeStamp start (grp->GetTimeStart());
  TTimeStamp end   (grp->GetTimeEnd());

  runinfo.start_time = start;
  runinfo.end_time = end;
  runinfo.cavern_pressure =
    grp->GetCavernAtmosPressure()->MakeGraph()->GetMean();
  //runinfo.surface_pressure = grp->GetSurfaceAtmosPressure();
  //runinfo.detector_mask = grp->GetDetectorMask();

  if (0) {
    TCanvas* c = new TCanvas("cavern pressure", "cavern pressure");
    grp->GetCavernAtmosPressure()->MakeGraph()->Draw("alp");
  }

  // -----------------------------------------------------------------------
  // HV data

  entry = man->Get("TRD/Calib/trd_hvAnodeUmon",run);
  AliTRDSensorArray* arr = (AliTRDSensorArray*)entry->GetObject();

  bool tmp = true;

  for (int i=0;i<arr->NumSensors();i++) {

    //cout << arr->GetSensorNum(i)->GetIdDCS() << "  "
    //<< arr->GetSensorNum(i)->GetStringID() << endl;

    //cout << arr->GetSensorNum(i)->Eval(start,true) << endl;;

    // somehow, this array does not produce a TGraph, therefore we use
    // the value at the start of the run
    runinfo.anode_voltage[i] = arr->GetSensorNum(i)->Eval(start,tmp);
  }

  entry = man->Get("TRD/Calib/trd_hvDriftUmon",run);
  arr = (AliTRDSensorArray*) entry->GetObject();

  for (int i=0;i<arr->NumSensors();i++) {
    // somehow, this array does not produce a TGraph, therefore we use
    // the value at the start of the run
    runinfo.drift_voltage[i] = arr->GetSensorNum(i)->Eval(start,tmp);
  }

  // get calibration information

  entry = man->Get("TRD/Calib/ChamberGainFactor",run);
  AliTRDCalDet* g = (AliTRDCalDet*)entry->GetObject();

  for (int i=0; i<540; i++) {
    runinfo.gain[i] = g->GetValue(i);
  }

  entry = man->Get("TRD/Calib/ChamberVdrift",run);
  AliTRDCalDet* vd = (AliTRDCalDet*)entry->GetObject();

  for (int i=0; i<540; i++) {
    runinfo.vdrift[i] = vd->GetValue(i);
  }

  entry = man->Get("TRD/Calib/ChamberExB",run);
  AliTRDCalDet* exb = (AliTRDCalDet*)entry->GetObject();

  for (int i=0; i<540; i++) {
    runinfo.ExB[i] = exb->GetValue(i);
  }


  cout << "      Start time : " << runinfo.start_time << endl;
  cout << "        End time : " << runinfo.end_time << endl;
  cout << " Cavern pressure : " << runinfo.cavern_pressure << endl;


  for (int i=0; i<10; i++) {
    cout << "chamber " << i << endl;
    cout << "      anode voltage : " << runinfo.anode_voltage[i] << endl;
    cout << "      drift voltage : " << runinfo.drift_voltage[i] << endl;
    cout << "               gain : " << runinfo.gain[i] << endl;
    cout << "     drift velocity : " << runinfo.vdrift[i] << endl;
    cout << "                ExB : " << runinfo.ExB[i] << endl;

    myfile << i << ", " << runinfo.gain[i] << std::endl;
  }

  myfile.close();
}
