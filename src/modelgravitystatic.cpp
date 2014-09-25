/***************************************************************************
*      Copyright (C) 2008 by Norwegian Computing Center and Statoil        *
***************************************************************************/

#include "src/definitions.h"
#include "src/modelgeneral.h"
#include "src/modelgravitystatic.h"
#include "src/xmlmodelfile.h"
#include "src/modelsettings.h"
#include "src/wavelet1D.h"
#include "src/wavelet3D.h"
#include "src/analyzelog.h"
#include "src/vario.h"
#include "src/simbox.h"
#include "src/background.h"
#include "src/welldata.h"
#include "src/blockedlogs.h"
#include "src/fftgrid.h"
#include "src/fftfilegrid.h"
#include "src/gridmapping.h"
#include "src/inputfiles.h"
#include "src/timings.h"
#include "src/io.h"
#include "src/waveletfilter.h"
#include "src/tasklist.h"
#include "src/parameteroutput.h"
#include "nrlib/surface/surface.hpp"

ModelGravityStatic::ModelGravityStatic(ModelSettings        *& modelSettings,
                                       ModelGeneral         *& modelGeneral,
                                       const InputFiles      * inputFiles)
{
  modelGeneral_           = modelGeneral;

  failed_                 = false;
  before_injection_start_ = false;

  bool failedLoadingModel = false;
  bool failedReadingFile  = false;
  std::string errText("");

  bool doGravityInversion = true;
  int numberGravityFiles  = 0;
  for(int i = 0; i<modelSettings->getNumberOfVintages(); i++){
    if(modelSettings->getGravityTimeLapse(i))
      numberGravityFiles++;
  }

  if(numberGravityFiles == 0){
    // Everything is ok - we do not need gravity inversion
    failedLoadingModel = false;
    doGravityInversion = false;
  }

  if(numberGravityFiles == 1){
    failedLoadingModel = true;
    doGravityInversion = false;
    errText+="Need at least two gravity surveys for inversion.";
  }

  // Set up gravimetric baseline
  if(doGravityInversion){

    LogKit::WriteHeader("Setting up gravimetric baseline");

    // Find first gravity data file
    std::string fileName = inputFiles->getGravimetricData(0);


    int nColumns = 5;  // We require data files to have five columns

    observation_location_utmx_.resize(1);
    observation_location_utmy_.resize(1);
    observation_location_depth_.resize(1);
    gravity_baseline_response_.resize(1);
    gravity_std_dev_.resize(1);

    ReadGravityDataFile(fileName, "gravimetric base survey",
                        nColumns,
                        observation_location_utmx_,
                        observation_location_utmy_,
                        observation_location_depth_,
                        gravity_baseline_response_,
                        gravity_std_dev_,
                        failedReadingFile,
                        errText);
    failedLoadingModel = failedReadingFile;

    //computeAdjustmentOfResponce(gravity_synt_initial_response_,gravity_synt_baseline_response_,modelGeneral->getState4D());


    Simbox * fullTimeSimbox = modelGeneral->getTimeSimbox();
    nxp_=modelSettings->getNXpad();
    nyp_=modelSettings->getNYpad();
    nzp_=modelSettings->getNZpad();



    x_upscaling_factor_ = modelSettings->getNXpad()/5 + 1;   // should be user input...
    y_upscaling_factor_ = modelSettings->getNYpad()/5 + 1;
    z_upscaling_factor_ = modelSettings->getNZpad()/5 + 1;

    SetUpscaledPaddingSize(modelSettings);  // NB: Changes upscaling factors!

    dx_upscaled_ = fullTimeSimbox->GetLX()/nx_upscaled_;
    dy_upscaled_ = fullTimeSimbox->GetLY()/ny_upscaled_;

    LogKit::LogFormatted(LogKit::Low, "Generating smoothing kernel ...");
    MakeUpscalingKernel(modelSettings, fullTimeSimbox);
    LogKit::LogFormatted(LogKit::Low, "ok.\n");

    LogKit::LogFormatted(LogKit::Low, "Generating lag index table (size " + NRLib::ToString(nxp_upscaled_) + " x "
                                                                          + NRLib::ToString(nyp_upscaled_) + " x "
                                                                          + NRLib::ToString(nzp_upscaled_) + ") ...");
    MakeLagIndex(nxp_upscaled_, nyp_upscaled_, nzp_upscaled_); // Including padded region!
    LogKit::LogFormatted(LogKit::Low, "ok.\n");
  }

  if (failedLoadingModel) {
    LogKit::WriteHeader("Error(s) with gravimetric surveys");
    LogKit::LogFormatted(LogKit::Error,"\n"+errText);
    LogKit::LogFormatted(LogKit::Error,"\nAborting\n");
  }

  failed_ = failedLoadingModel || failedReadingFile;
  failed_details_.push_back(failedReadingFile);
}


ModelGravityStatic::~ModelGravityStatic(void)
{
}

void
ModelGravityStatic::computeBaseAdjustments(ModelGeneral *modelGeneral)
{
  LogKit::WriteHeader("Computing base adjustments to initial time for gravimetric data");
  double gamma = 6.67384e-11; // units: m^3/(kg*s^2)
  int nObs=static_cast<int>(gravity_baseline_response_.size());
  gravity_synt_initial_response_.resize(nObs,0.0f);
  gravity_synt_baseline_response_.resize(nObs,0.0f);
  gravity_initial_response_.resize(nObs,0.0f);

  State4D* state4D = modelGeneral->getState4D();
  FFTGrid * rhoInitial = new FFTGrid( state4D->getMuRhoStatic() );
  FFTGrid * rhoBase    = new FFTGrid( state4D->getMuRhoDynamic());
  rhoInitial->invFFTInPlace();
  rhoBase->invFFTInPlace();
  rhoBase->add(rhoInitial);
  rhoInitial->expTransf();
  rhoBase->expTransf();

  Simbox * fullSizeDepthSimbox = modelGeneral_->getDepthSimbox();
  int nx = fullSizeDepthSimbox->getnx();
  int ny = fullSizeDepthSimbox->getny();
  int nz = fullSizeDepthSimbox->getnz();

  double dx = fullSizeDepthSimbox->getdx();
  double dy = fullSizeDepthSimbox->getdy();


  for(int i = 0; i < nObs; i++){
    double x0 = observation_location_utmx_[i];
    double y0 = observation_location_utmy_[i];
    double z0 = observation_location_depth_[i];
    for(int ii = 0; ii < nx; ii++){
      for(int jj = 0; jj < ny; jj++){
        for(int kk = 0; kk < nz; kk++){
          double x, y, z;
          fullSizeDepthSimbox->getCoord(ii, jj, kk, x, y, z); // assuming these are center positions...

          double dz = fullSizeDepthSimbox->getdz(ii, jj);
          double rI=rhoInitial->getRealValue(ii,jj,kk);// units tonn/m^3 =1000 kg/m^3
          double rB=rhoBase->getRealValue(ii,jj,kk);  // units tonn/m^3 =1000 kg/m^3
          double localVolume = dx*dy*dz; // units m^3
          double localDistanceSquared = (x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0); //units m^2
          gravity_synt_initial_response_[i] +=  rI*localVolume/localDistanceSquared;
          gravity_synt_baseline_response_[i] += rB*localVolume/localDistanceSquared;
        }
      }
    }
    gravity_synt_initial_response_[i]  *=gamma*1000*1e8;// [1000 is tonn to kg]  1e8 is m/s^2  to mikro gal
    gravity_synt_baseline_response_[i] *=gamma*1000*1e8;// [1000 is tonn to kg]  1e8 is m/s^2  to mikro gal
    gravity_initial_response_[i]        = static_cast<float>(gravity_synt_initial_response_[i]-gravity_synt_baseline_response_[i]+gravity_baseline_response_[i]);
  }
}

void
ModelGravityStatic::ReadGravityDataFile(const std::string   & fileName,
                                        const std::string   & readReason,
                                        int                   nColumns,
                                        std::vector <float> & obs_loc_utmx,
                                        std::vector <float> & obs_loc_utmy,
                                        std::vector <float> & obs_loc_depth,
                                        std::vector <float> & gravity_response,
                                        std::vector <float> & gravity_std_dev,
                                        bool                  failed,
                                        std::string         & errText)
{
  int nObsMax=300;
  float * tmpRes = new float[nObsMax*nColumns+1];
  std::ifstream inFile;
  NRLib::OpenRead(inFile,fileName);
  std::string text = "Reading "+readReason+" from file "+fileName+" ... ";
  LogKit::LogFormatted(LogKit::Low,text);
  std::string storage;
  int index = 0;
  failed = false;
  int line_num = 1;

  while(failed == false && (NRLib::CheckEndOfFile(inFile)==false)) {
      NRLib::ReadNextToken(inFile ,storage, line_num);
      try {
        tmpRes[index] = NRLib::ParseType<float>(storage);
      }
      catch (NRLib::Exception & e) {
        errText += "Error in "+fileName+"\n";
        errText += e.what();
        failed = true;
      }
    index++;
  }
   int nObs = index/nColumns; // integer division

  if(failed == false) {
    if(index != nObs*nColumns) {
      failed = true;
      errText += "Found "+NRLib::ToString(index)+" numbers in file "+fileName+", expected "+NRLib::ToString(nObs*nColumns)+".\n";
    }
  }

  if(failed == false) {
    LogKit::LogFormatted(LogKit::Low,"ok.\n");
    obs_loc_utmx.resize(nObs);
    obs_loc_utmy.resize(nObs);
    obs_loc_depth.resize(nObs);
    gravity_response.resize(nObs);
    gravity_std_dev.resize(nObs);
    index = 0;
    for(int i=0;i<nObs;i++) {
      obs_loc_utmx[i] = tmpRes[index];
      index++;
      obs_loc_utmy[i] = tmpRes[index];
      index++;
      obs_loc_depth[i] = tmpRes[index];
      index++;
      gravity_response[i] = tmpRes[index];
      index++;
      gravity_std_dev[i] = tmpRes[index];
      index++;
    }
  }
  else
    LogKit::LogFormatted(LogKit::Low,"failed.\n");

  delete [] tmpRes;
}

void
ModelGravityStatic::MakeUpscalingKernel(ModelSettings * modelSettings, Simbox * fullTimeSimbox)
{ //
  int nx = fullTimeSimbox->getnx();
  int ny = fullTimeSimbox->getny();
  int nz = fullTimeSimbox->getnz();

  int nxp = modelSettings->getNXpad();
  int nyp = modelSettings->getNYpad();
  int nzp = modelSettings->getNZpad();

  upscaling_kernel_ = new FFTGrid(nx, ny, nz, nxp, nyp, nzp);
  upscaling_kernel_->setType(FFTGrid::OPERATOR);
  upscaling_kernel_->fillInConstant(0.0);

  upscaling_kernel_->setAccessMode(FFTGrid::RANDOMACCESS);

  for(int k = 0; k < nzp/nz_upscaled_; k++)
    for(int j = 0; j < nyp/ny_upscaled_; j++)
      for(int i = 0; i < nxp/nx_upscaled_; i++)
        upscaling_kernel_->setRealValue(i, j, k, 1.0,true);

  upscaling_kernel_->endAccess();

  upscaling_kernel_->multiplyByScalar(static_cast<float>(nxp_upscaled_*nyp_upscaled_*nzp_upscaled_)/static_cast<float>(nxp*nyp*nzp));
  upscaling_kernel_->fftInPlace();
}

void ModelGravityStatic::MakeLagIndex(int nx_upscaled, int ny_upscaled, int nz_upscaled)
{
  int N_up = nx_upscaled*ny_upscaled*nz_upscaled;
  lag_index_.resize(N_up);
  for (int i = 0; i < N_up; ++i) {
    lag_index_[i].resize(N_up);

    for (int j = 0; j < N_up; ++j)
      lag_index_[i][j].resize(3);
  }

  int I, J;
  for(int k1 = 1; k1 <= nz_upscaled; k1++)
    for(int j1 = 1; j1 <= ny_upscaled; j1++)
      for(int i1 = 1; i1 <= nx_upscaled; i1++){
        I =  i1 + (j1-1)*nx_upscaled + (k1-1)*nx_upscaled*ny_upscaled;

        for(int k2 = 1; k2 <= nz_upscaled; k2++)
          for(int j2 = 1; j2 <= ny_upscaled; j2++)
            for(int i2 = 1; i2 <= nx_upscaled; i2++){
              J = i2 + (j2-1)*nx_upscaled + (k2-1)*nx_upscaled*ny_upscaled;

              int lag_i = i2 - i1;
              int lag_j = j2 - j1;
              int lag_k = k2 - k1;

              int ind1, ind2, ind3;
              if(abs(lag_i) <= nx_upscaled/2 && abs(lag_j) <= ny_upscaled/2 && abs(lag_k) <= nz_upscaled/2) {
                if(lag_i >= 0)
                  ind1 = lag_i + 1;
                else
                  ind1 = nx_upscaled + lag_i + 1;

                if(lag_j >= 0)
                  ind2 = lag_j + 1;
                else
                  ind2 = ny_upscaled + lag_j + 1;

                if(lag_k >= 0)
                  ind3 = lag_k + 1;
                else
                  ind3 = nz_upscaled + lag_k + 1;

                lag_index_[I-1][J-1][0] = ind1 - 1;   // NB: -1
                lag_index_[I-1][J-1][1] = ind2 - 1;
                lag_index_[I-1][J-1][2] = ind3 - 1;
              }
              else
              {
                lag_index_[I-1][J-1][0] = -1;
                lag_index_[I-1][J-1][1] = -1;
                lag_index_[I-1][J-1][2] = -1;
              }
            }
      }
}
void
ModelGravityStatic::SetUpscaledPaddingSize(ModelSettings * modelSettings)
{
  // Find original nxp, nyp, nzp
  int nxpad = modelSettings->getNXpad();
  int nypad = modelSettings->getNYpad();
  int nzpad = modelSettings->getNZpad();

  int nxpad_up = SetPaddingSize(nxpad, x_upscaling_factor_);
  int nypad_up = SetPaddingSize(nypad, y_upscaling_factor_);
  int nzpad_up = SetPaddingSize(nzpad, z_upscaling_factor_);

  // Initilizing!
  nxp_upscaled_ = nxpad_up;
  nyp_upscaled_ = nypad_up;
  nzp_upscaled_ = nzpad_up;

  nx_upscaled_ = nxpad_up;
  ny_upscaled_ = nypad_up;
  nz_upscaled_ = nzpad_up;

  // Set true upscaling factors
  x_upscaling_factor_ = nxpad/nxp_upscaled_;
  y_upscaling_factor_ = nypad/nyp_upscaled_;
  z_upscaling_factor_ = nzpad/nzp_upscaled_;
}


int
ModelGravityStatic::SetPaddingSize(int original_nxp, int upscaling_factor)
{
  int leastint = static_cast<int>(ceil(static_cast<double>(original_nxp)/static_cast<double>(upscaling_factor)));

  std::vector<int> exp_list = findClosestFactorableNumber(original_nxp);

  int closestprod = original_nxp;

  int factor   =       1;

  for(int i=0;i<exp_list[0]+1;i++)
    for(int j=0;j<exp_list[1]+1;j++)
      for(int k=0;k<exp_list[2]+1;k++)
        for(int l=0;l<exp_list[3]+1;l++)
          for(int m=0;m<exp_list[4]+1;m++)
            for(int n=exp_list[4];n<exp_list[5]+1;n++)
            {
              factor = static_cast<int>(pow(2.0f,i)*pow(3.0f,j)*pow(5.0f,k)*
                pow(7.0f,l)*pow(11.0f,m)*pow(13.0f,n));
              if ((factor >=  leastint) &&  (factor <  closestprod))
              {
                closestprod=factor;
              }
            }
            return closestprod;
}

// Same as in FFTGrid-class, however, this one returns list of exponents
std::vector<int> ModelGravityStatic::findClosestFactorableNumber(int leastint)
{
  int i,j,k,l,m,n;
  int factor   =       1;

  std::vector<int> exp_list(6);

  int maxant2    = static_cast<int>(ceil(static_cast<double>(log(static_cast<float>(leastint))) / log(2.0f) ));
  int maxant3    = static_cast<int>(ceil(static_cast<double>(log(static_cast<float>(leastint))) / log(3.0f) ));
  int maxant5    = static_cast<int>(ceil(static_cast<double>(log(static_cast<float>(leastint))) / log(5.0f) ));
  int maxant7    = static_cast<int>(ceil(static_cast<double>(log(static_cast<float>(leastint))) / log(7.0f) ));
  int maxant11   = 0;
  int maxant13   = 0;

  int closestprod= static_cast<int>(pow(2.0f,maxant2));
  exp_list[0] = maxant2;
  exp_list[1] = 0;
  exp_list[2] = 0;
  exp_list[3] = 0;
  exp_list[4] = 0;
  exp_list[5] = 0;

  for(i=0;i<maxant2+1;i++)
    for(j=0;j<maxant3+1;j++)
      for(k=0;k<maxant5+1;k++)
        for(l=0;l<maxant7+1;l++)
          for(m=0;m<maxant11+1;m++)
            for(n=maxant11;n<maxant13+1;n++)
            {
              factor = static_cast<int>(pow(2.0f,i)*pow(3.0f,j)*pow(5.0f,k)*
                pow(7.0f,l)*pow(11.0f,m)*pow(13.0f,n));
              if ((factor >=  leastint) &&  (factor <  closestprod))
              {
                exp_list[0] = i;
                exp_list[1] = j;
                exp_list[2] = k;
                exp_list[3] = l;
                exp_list[4] = m;
                exp_list[5] = n;
                closestprod=factor;
              }
            }
  return exp_list;
}
