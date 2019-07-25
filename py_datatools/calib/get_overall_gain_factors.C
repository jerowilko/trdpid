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

float combined_gain_factor[540][16][144];
float chamber_gain_factor[540];

//__________________________________________________________________________
void get_overall_gain_factors(Int_t year=2016, Int_t run=265377)
{
  // set up the connection to the OCDB.
  AliCDBManager* man = AliCDBManager::Instance();
  AliTRDcalibDB* const calibration = AliTRDcalibDB::Instance();

  man->SetDefaultStorage(Form("alien://folder=/alice/data/%d/OCDB/",year));

  man->SetCacheFlag(kTRUE);
  man->SetRun(run);


  entry = man->Get("TRD/Calib/ChamberGainFactor",run);
  AliTRDCalDet* g = (AliTRDCalDet*)entry->GetObject();



  // Get the chamber level gain factors.
  for (int det=0; det<540; det++) {
    float chamber_gain = g->GetValue(det);
    chamber_gain_factor[det] = chamber_gain;
  }



  // Get the MCM online local gain factors and the offline local gain factors
  entry = man->Get("TRD/Calib/LocalGainFactor", run);
  AliTRDCalPad* loc = (AliTRDCalPad*)entry->GetObject();

  for (int det=0; det<540; det++) {
    AliTRDCalROC *det_cal_roc = loc->GetCalROC(det);
    fCalOnlGainROC = calibration->GetOnlineGainTableROC(det);

    if (det < 5) cout << "\n\n" << "det: " << det;
    for (int r=0; r<16; r++) {
        if (det < 5 && r < 5) cout << "\n";
        for (int c=0; c<144; c++) {

            // Initialize to 1.0 in case any info is missing.
            float online_local_gain_factor = 1.0;
            float local_gain_factor = 1.0;

            // Some chambers are smaller than others
            if (r < det_cal_roc->GetNrows()) {
                local_gain_factor = det_cal_roc->GetValue(c, r);

                // According to some docs, no MCM correction was used in some chambers, hence we must check whether fCalOnlGainROC is a null pointer.
                // If it is a null pointer, then the gain factor remains at 1.0.
                if (fCalOnlGainROC) online_local_gain_factor = fCalOnlGainROC->GetGainCorrectionFactor(r, c);
            }

            // Combine all the gain factors for each pad
            combined_gain_factor[det][r][c] = chamber_gain_factor[det] * local_gain_factor / online_local_gain_factor;

            // Digits are then updated as tracklet[det][r][c] *= combined_gain_factor[det][r][c];

            if (det < 5 && r < 5 && c < 5) cout << combined_gain_factor[det][r][c] << ", ";
        }
    }
  }
}
