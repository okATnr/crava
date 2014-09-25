/***************************************************************************
*      Copyright (C) 2008 by Norwegian Computing Center and Statoil        *
***************************************************************************/

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <limits.h>
#define _USE_MATH_DEFINES
#include <cmath>

#include "src/definitions.h"
#include "src/modelgeneral.h"
#include "src/modelgravitystatic.h"
#include "src/modelgravitydynamic.h"
#include "src/gravimetricinversion.h"
#include "src/xmlmodelfile.h"
#include "src/modelsettings.h"
#include "src/simbox.h"
#include "src/background.h"
#include "src/fftgrid.h"
#include "src/fftfilegrid.h"
#include "src/gridmapping.h"
#include "src/inputfiles.h"
#include "src/timings.h"
#include "src/io.h"
#include "src/tasklist.h"
#include "src/seismicparametersholder.h"

#include "lib/utils.h"
#include "lib/random.h"
#include "lib/timekit.hpp"
#include "nrlib/iotools/fileio.hpp"
#include "nrlib/iotools/stringtools.hpp"
#include "nrlib/segy/segy.hpp"
#include "nrlib/iotools/logkit.hpp"
#include "nrlib/stormgrid/stormcontgrid.hpp"
#include "nrlib/volume/volume.hpp"


ModelGravityDynamic::ModelGravityDynamic(const ModelSettings          * modelSettings,
                                         const ModelGeneral           * modelGeneral,
                                         ModelGravityStatic           * modelGravityStatic,
                                         const InputFiles             * inputFiles,
                                         int                            t)

{
  modelGeneral_ = modelGeneral;

  failed_                 = false;
  thisTimeLapse_          = t;

  bool failedLoadingModel = false;
  bool failedReadingFile  = false;
  std::string errText("");

  int nColumns = 5;  // We require data files to have five columns

  // Check that timeLapse is ok
  if(thisTimeLapse_ < 1 && thisTimeLapse_ >modelSettings->getNumberOfVintages()){
    errText += "Not valid time lapse";
    failedLoadingModel = true;
  }

  if(failedLoadingModel == false){
    LogKit::WriteHeader("Setting up gravimetric time lapse");

    // Find first gravity data file
    std::string fileName = inputFiles->getGravimetricData(thisTimeLapse_);

    observation_location_utmx_ .resize(1);
    observation_location_utmy_ .resize(1);
    observation_location_depth_.resize(1);
    gravity_response_.resize(1);
    gravity_std_dev_ .resize(1);

    ModelGravityStatic::ReadGravityDataFile(fileName,
                                            "gravimetric survey ",
                                            nColumns,
                                            observation_location_utmx_,
                                            observation_location_utmy_,
                                            observation_location_depth_,
                                            gravity_response_,
                                            gravity_std_dev_,
                                            failedReadingFile,
                                            errText);
    failedLoadingModel = failedReadingFile;

    LogKit::LogFormatted(LogKit::Low, "Setting up forward model matrix ...");
    BuildGMatrix(modelGravityStatic);
    LogKit::LogFormatted(LogKit::Low, "ok.\n");
  }

  if (failedLoadingModel) {
    LogKit::WriteHeader("Error(s) with gravimetric surveys.");
    LogKit::LogFormatted(LogKit::Error,"\n"+errText);
    LogKit::LogFormatted(LogKit::Error,"\nAborting\n");
  }

  failed_ = failedLoadingModel || failedReadingFile;
  failed_details_.push_back(failedReadingFile);

  }

ModelGravityDynamic::~ModelGravityDynamic(void)
{
}

void ModelGravityDynamic::BuildGMatrix(ModelGravityStatic      * modelGravityStatic)
{
  // Building gravity matrix for each time vintage, using updated mean Vp in generating the grid.
  double gamma = 6.67384e-11; // units: m^3/(kg*s^2)

 // Simbox * fullSizeTimeSimbox = modelGeneral_->getTimeSimbox();
  Simbox * fullSizeDepthSimbox = modelGeneral_->getDepthSimbox();

  // Use vp_current, found in Seismic parameters holder.
 // FFTGrid * expMeanAlpha      = new FFTGrid(seismicParameters.GetMuAlpha());  // for upscaling
 // FFTGrid * meanAlphaFullSize = new FFTGrid(expMeanAlpha);                    // for full size matrix

  int nx = fullSizeDepthSimbox->getnx();
  int ny = fullSizeDepthSimbox->getny();
  int nz = fullSizeDepthSimbox->getnz();

  double dx = fullSizeDepthSimbox->getdx();
  double dy = fullSizeDepthSimbox->getdy();

  int nxp = modelGravityStatic->GetNxp();
  int nyp = modelGravityStatic->GetNyp();
  int nzp = modelGravityStatic->GetNzp();

  int nxp_upscaled = modelGravityStatic->GetNxp_upscaled();
  int nyp_upscaled = modelGravityStatic->GetNyp_upscaled();
  int nzp_upscaled = modelGravityStatic->GetNzp_upscaled();

  int upscaling_factor_x = nxp/nxp_upscaled;
  int upscaling_factor_y = nyp/nyp_upscaled;
  int upscaling_factor_z = nzp/nzp_upscaled;

  // dimensions of one grid cell
  double dx_upscaled = dx*upscaling_factor_x;
  double dy_upscaled = dy*upscaling_factor_y;

  int nx_upscaled = modelGravityStatic->GetNx_upscaled();
  int ny_upscaled = modelGravityStatic->GetNy_upscaled();
  int nz_upscaled = modelGravityStatic->GetNz_upscaled();

  int N_upscaled = nx_upscaled*ny_upscaled*nz_upscaled;
  int N_fullsize = nx*ny*nz;

  int nObs = modelGravityStatic->GetNData();

  G_         .resize(nObs, N_upscaled);
  G_fullsize_.resize(nObs, N_fullsize);

  // Need to be in real domain for transforming from log domain
 // if(expMeanAlpha->getIsTransformed())
 //   expMeanAlpha->invFFTInPlace();

 // if(meanAlphaFullSize->getIsTransformed())
 //   meanAlphaFullSize->invFFTInPlace();

 // float sigma_squared = GravimetricInversion::GetSigmaForTransformation(seismicParameters.GetCovAlpha());
 // GravimetricInversion::MeanExpTransform(expMeanAlpha,      sigma_squared);
 // GravimetricInversion::MeanExpTransform(meanAlphaFullSize, sigma_squared);


  //Smooth (convolve) and subsample
 // FFTGrid * upscalingKernel_conj = modelGravityStatic->GetUpscalingKernel();
 //  if(upscalingKernel_conj->getIsTransformed() == false)
 //   upscalingKernel_conj->fftInPlace();
 // upscalingKernel_conj->conjugate();  // Conjugate only in FFT domain.

  // Need to be in FFT domain for convolution and subsampling
 // if(expMeanAlpha->getIsTransformed() == false)
  //  expMeanAlpha->fftInPlace();

 // expMeanAlpha->multiply(upscalingKernel_conj);  // Now is expMeanAlpha smoothed

  //FFTGrid * upscaledMeanAlpha;
  //GravimetricInversion::Subsample(upscaledMeanAlpha, expMeanAlpha, nx_upscaled, ny_upscaled, nz_upscaled, nxp_upscaled, nyp_upscaled, nzp_upscaled);

 // upscaledMeanAlpha->invFFTInPlace();

  float x0, y0, z0; // Coordinates for the observations points
  int J = 0;        // Index in matrix counting cell number
  int I = 0;

 // float  vp;
  //double dt;
  double dz;
  double localMass;
  double localDistanceSquared;

  for(int i = 0; i < nObs; i++){
    x0 = observation_location_utmx_[i];
    y0 = observation_location_utmy_[i];
    z0 = observation_location_depth_[i];

     J = 0; I = 0;

    // Loop through upscaled simbox to get x, y, z for each grid cell
    for(int ii = 0; ii < nx_upscaled; ii++){
      for(int jj = 0; jj < ny_upscaled; jj++){
        for(int kk = 0; kk < nz_upscaled; kk++){
          double x, y, z;
          //vp = upscaledMeanAlpha->getRealValue(ii,jj,kk);
          int istart = ii*upscaling_factor_x;
          int istop  = (ii+1)*upscaling_factor_x;
          int jstart = jj*upscaling_factor_y;
          int jstop  = (jj+1)*upscaling_factor_y;
          int kstart = kk*upscaling_factor_z;
          int kstop = (kk+1)*upscaling_factor_z;

       //   double x_local  = 0;
       //   double y_local  = 0;
          //double z_local  = 0;
          //double dt_local = 0;
          double dz_local = 0;
          double squaredDist=0;
          int counterIJ=0;
          int counterIJK=0;
          //Find center position of coarse grid cell using indices of fine grid and averaging their cell centers.
          for(int iii=istart; iii< std::min(nx,istop) ; iii++){
            for(int jjj=jstart; jjj<std::min(ny,jstop); jjj++){
              counterIJ++;
              for(int kkk=kstart; kkk<std::min(nz,kstop); kkk++){
                fullSizeDepthSimbox->getCoord(iii, jjj, kkk, x, y, z);
               // x_local += x;
               // y_local += y;
               // z_local += z;
                squaredDist += (x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0);
                //dt_local += fullSizeTimeSimbox->getdz(iii, jjj);
                dz_local += fullSizeDepthSimbox->getdz(iii, jjj);
                counterIJK++;
              }
            }
          }
          double zfactor=1;
          if(counterIJK > 0){
           // x_local  /= counterIJK;  //average => cell center
           // y_local  /= counterIJK;  // average=> cell center
           // z_local  /= counterIJK;  // average=> cell center
            squaredDist/=counterIJK; // average squared distance  in cell
           // dt_local /= counterIJ; // sum in vertical direction average in lateral
            dz_local /= counterIJ; // sum in vertical direction average in lateral
          }else{ // We are outside the reservoir region. Set dummyvalues and assure that there is no contribution in G
           // x_local = x0+1;y_local = y0+1; z_local = z0+1000;dt_local=1;
            dz_local=1;
            squaredDist= 1000*1000;
            zfactor=0;
          }
          // Find fraction of dx_upscaled and dy_upscaled according to indicies
          double xfactor = 1;
          if(istop <= nx){  // inside
            xfactor = 1;
          }
          else if(istart > nx){ //outside
            xfactor = 0;
          }
          else{
            xfactor = static_cast<double>(nx - istart)/static_cast<double>(upscaling_factor_x);
            //x_local -= (1-xfactor)*dx_upscaled*0.5; // move effective cell center when cell contains padding
          }

          double yfactor = 1;
          if(jstop <= ny){
            yfactor = 1;
          }
          else if(jstart > ny){
            yfactor = 0;
          }
          else{
            yfactor = static_cast<double>(ny - jstart)/static_cast<double>(upscaling_factor_y);
            //y_local -= (1-yfactor)*dy_upscaled*0.5;// move effective cell center when cell contains padding
          }
          //double  localMass2 = (xfactor*dx_upscaled)*(yfactor*dy_upscaled)*(dt_local/1000*zfactor)*(vp*0.5); //m x m x s x (m/s) = m^3 gives  [units kg when multiplied with density kg/m^3 ]
          localMass = (xfactor*dx_upscaled)*(yfactor*dy_upscaled)*(dz_local*zfactor);
          //localDistanceSquared = (x_local-x0)*(x_local-x0) + (y_local-y0)*(y_local-y0) + (z_local-z0)*(z_local-z0); //units m^2
          G_(i,J) = localMass/squaredDist;
          J++;
        }
      }
    }

    // Loop through full size simbox to get x, y, z for each grid cell
    for(int ii = 0; ii < nx; ii++){
      for(int jj = 0; jj < ny; jj++){
        for(int kk = 0; kk < nz; kk++){
          double x, y, z;
          fullSizeDepthSimbox->getCoord(ii, jj, kk, x, y, z); // assuming these are center positions...
          //vp = meanAlphaFullSize->getRealValue(ii, jj, kk);
          //dt = fullSizeTimeSimbox->getdz(ii, jj);
          dz = fullSizeDepthSimbox->getdz(ii, jj);
         // double localMass2 = dx*dy*(dt/1000)*(vp*0.5); // units kg
          double localMass = dx*dy*(dz); // units kg
          localDistanceSquared = pow((x-x0),2) + pow((y-y0),2) + pow((z-z0),2); //units m^2
          G_fullsize_(i,I) = localMass/localDistanceSquared;
          I++;
        }
      }
    }
  }

  // Units G is now [m= m^3/m^2 ] G*m has units [m*tonn/m^3 =1000 kg/m^2] // crava unit for density is   tonn/m^3 or g/ccm
  // gamma has units: m^3/(kg*s^2)
  //  G*m*gamma has units  [1000 kg/m^2 * m^3/(kg*s^2)  = 1000 m/s^2 ]
  // we want  mikro gal as units

  G_ = G_*(gamma*1000*1e8);
  G_fullsize_ = G_fullsize_*(gamma*1000*1e8); // [1000 is tonn to kg]  1e8 is m/s^2  to mikro gal
  NRLib::WriteMatrixToFile("G_fullsize.dat",G_fullsize_);
  NRLib::WriteMatrixToFile("G_upscaled.dat",G_);
}
