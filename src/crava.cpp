/***************************************************************************
*      Copyright (C) 2008 by Norwegian Computing Center and Statoil        *
***************************************************************************/

#include "rfftw.h"

#ifdef PARALLEL
#include <omp.h>
#endif

#include "src/crava.h"
#include "src/wavelet.h"
#include "src/wavelet1D.h"
#include "src/wavelet3D.h"
#include "src/modelgeneral.h"
#include "src/modelavostatic.h"
#include "src/modelavodynamic.h"
#include "src/fftgrid.h"
#include "src/fftfilegrid.h"
#include "src/vario.h"
#include "src/welldata.h"
#include "src/krigingdata3d.h"
#include "src/covgridseparated.h"
#include "src/krigingadmin.h"
#include "src/faciesprob.h"
#include "src/definitions.h"
#include "src/gridmapping.h"
#include "src/parameteroutput.h"
#include "src/timings.h"
#include "src/spatialwellfilter.h"
#include "src/qualitygrid.h"
#include "src/io.h"
#include "src/tasklist.h"

#include "lib/timekit.hpp"
#include "lib/random.h"
#include "lib/lib_matr.h"

#include "nrlib/iotools/logkit.hpp"
#include "nrlib/stormgrid/stormcontgrid.hpp"
#include "nrlib/grid/grid2d.hpp"
#include "rplib/distributionsstoragekit.h"
#include "rplib/distributionsrock.h"

#include "nrlib/flens/nrlib_flens.hpp"

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <time.h>
#include <string>

Crava::Crava(ModelSettings           * modelSettings,
             ModelGeneral            * modelGeneral,
             ModelAVOStatic          * modelAVOstatic,
             ModelAVODynamic         * modelAVOdynamic,
             SeismicParametersHolder & seismicParameters)
{
  if(modelSettings->getForwardModeling())
    LogKit::LogFormatted(LogKit::Low,"\nBuilding model ...\n");

  LogKit::WriteHeader("Building Stochastic Model");

  time_t timestart, timeend;
  time(&timestart);

  double wall=0.0, cpu=0.0;
  TimeKit::getTime(wall,cpu);

  modelSettings_     = modelSettings;
  modelGeneral_      = modelGeneral;
  modelAVOstatic_    = modelAVOstatic;
  modelAVOdynamic_   = modelAVOdynamic;

  nx_                = seismicParameters.GetMuAlpha()->getNx();
  ny_                = seismicParameters.GetMuAlpha()->getNy();
  nz_                = seismicParameters.GetMuAlpha()->getNz();
  nxp_               = seismicParameters.GetMuAlpha()->getNxp();
  nyp_               = seismicParameters.GetMuAlpha()->getNyp();
  nzp_               = seismicParameters.GetMuAlpha()->getNzp();
  lowCut_            = modelSettings_->getLowCut();
  highCut_           = modelSettings_->getHighCut();
  wnc_               = modelSettings_->getWNC();     // white noise component see crava.h
  energyTreshold_    = modelSettings_->getEnergyThreshold();
  ntheta_            = modelAVOdynamic->getNumberOfAngles();
  doing4DInversion_  = modelSettings->getDo4DInversion();
  fileGrid_          = modelSettings_->getFileGrid();
  outputGridsSeismic_= modelSettings_->getOutputGridsSeismic();
  outputGridsElastic_= modelSettings_->getOutputGridsElastic();
  writePrediction_   = modelSettings_->getWritePrediction();
  krigingParameter_  = modelSettings_->getKrigingParameter();
  nWells_            = modelSettings_->getNumberOfWells();
  nSim_              = modelSettings_->getNumberOfSimulations();
  wells_             = modelGeneral_->getWells();
  simbox_            = modelGeneral_->getTimeSimbox();
  random_            = modelGeneral_->getRandomGen();
  seisWavelet_       = modelAVOdynamic_->getWavelets();
  A_                 = modelAVOdynamic_->getAMatrix();
  meanAlpha_         = seismicParameters.GetMuAlpha();
  meanBeta_          = seismicParameters.GetMuBeta();
  meanRho_           = seismicParameters.GetMuRho();
  postAlpha_         = copyFFTGrid(meanAlpha_);
  postBeta_          = copyFFTGrid(meanBeta_);
  postRho_           = copyFFTGrid(meanRho_);
  fprob_             = NULL;
  thetaDeg_          = new float[ntheta_];
  empSNRatio_        = new float[ntheta_];
  theoSNRatio_       = new float[ntheta_];
  modelVariance_     = new float[ntheta_];
  signalVariance_    = new float[ntheta_];
  errorVariance_     = new float[ntheta_];
  dataVariance_      = new float[ntheta_];
  scaleWarning_      = 0;
  scaleWarningText_  = "";
  errThetaCov_       = new double*[ntheta_];
  sigmamdnew_        = NULL;
  errCorr_           = NULL;

  for(int i=0;i<ntheta_;i++) {
    errThetaCov_[i]  = new double[ntheta_];
    thetaDeg_[i]     = static_cast<float>(modelAVOdynamic_->getAngle(i)*180.0/NRLib::Pi);
  }

  SpatialWellFilter * spatwellfilter = NULL;

  // reality check: all dimensions involved match
  assert(meanBeta_->consistentSize(nx_,ny_,nz_,nxp_,nyp_,nzp_));
  assert(meanRho_->consistentSize(nx_,ny_,nz_,nxp_,nyp_,nzp_));

  if(!modelSettings_->getForwardModeling()) {
    priorVar0_      = seismicParameters.getPriorVar0();
    seisData_       = modelAVOdynamic_->getSeisCubes();
    modelAVOdynamic_->releaseGrids();

    if (modelSettings->getDoInversion() && spatwellfilter == NULL) {
      spatwellfilter = new SpatialWellFilter(modelSettings->getNumberOfWells());

      FFTGrid * alphaCov = seismicParameters.GetCovAlpha();
      alphaCov->setAccessMode(FFTGrid::RANDOMACCESS);

      for(int i=0; i<nWells_; i++)
        spatwellfilter->setPriorSpatialCorr(alphaCov, wells_[i], i);

      alphaCov->endAccess();
    }

    float corrGradI, corrGradJ;
    modelGeneral_->getCorrGradIJ(corrGradI, corrGradJ);

    fftw_real * corrT = seismicParameters.extractParamCorrFromCovAlpha(nzp_);

    float dt = static_cast<float>(modelGeneral->getTimeSimbox()->getdz());
    if((modelSettings_->getOtherOutputFlag() & IO::PRIORCORRELATIONS) > 0)
      seismicParameters.writeFilePriorCorrT(corrT, nzp_, dt);

    errCorr_ = createFFTGrid();
    errCorr_ ->setType(FFTGrid::COVARIANCE);
    errCorr_ ->createRealGrid();
    errCorr_->fillInErrCorr(modelGeneral->getPriorCorrXY(), corrGradI, corrGradJ); // errCorr_->fftInPlace();

    for(int i=0 ; i< ntheta_ ; i++)
      assert(seisData_[i]->consistentSize(nx_,ny_,nz_,nxp_,nyp_,nzp_));

    computeVariances(corrT, modelSettings_);
    scaleWarning_ = checkScale();  // fills in scaleWarningText_ if needed.
    fftw_free(corrT);

    if((modelSettings->getOtherOutputFlag() & IO::PRIORCORRELATIONS) > 0) {
      float * corrTFiltered = seismicParameters.getPriorCorrTFiltered(nz_, nzp_);
      seismicParameters.writeFilePriorCorrT(corrTFiltered, nzp_, dt);     // No zeros in the middle
      delete [] corrTFiltered;
    }

    if(simbox_->getIsConstantThick() == false)
      divideDataByScaleWavelet(seismicParameters);

    for(int i = 0 ; i < ntheta_ ; i++){
      seisData_[i]->setAccessMode(FFTGrid::RANDOMACCESS);
      seisData_[i]->fftInPlace();
      seisData_[i]->endAccess();
    }
    postAlpha_->fftInPlace();
    postBeta_ ->fftInPlace();
    postRho_  ->fftInPlace();
  }
  else{
     LogKit::WriteHeader("Mark 4");
    modelAVOdynamic_->releaseGrids();
  }

  Timings::setTimeStochasticModel(wall,cpu);

  if(!modelSettings->getForwardModeling()){
    if(scaleWarning_ != 0){
      LogKit::LogFormatted(LogKit::Low,"\nWarning  !!!\n");
      LogKit::LogFormatted(LogKit::Low,"%s",scaleWarningText_.c_str());
      LogKit::LogFormatted(LogKit::Low,"\n");
    }

    printEnergyToScreen();

    time(&timeend);
    LogKit::LogFormatted(LogKit::DebugLow,"\nTime elapsed :  %d\n",timeend-timestart);

    computePostMeanResidAndFFTCov(modelGeneral_,
                                  simbox_,
                                  seismicParameters,
                                  postAlpha_,
                                  postBeta_,
                                  postRho_,
                                  errCorr_,
                                  seisData_,
                                  seisWavelet_,
                                  errThetaCov_,
                                  A_,
                                  thetaDeg_,
                                  ntheta_,
                                  lowCut_,
                                  highCut_,
                                  wnc_,
                                  nz_,
                                  nzp_,
                                  nyp_,
                                  nxp_,
                                  modelSettings->getNumberOfThreads(),
                                  fileGrid_);

  //Finish use of seisData, since we need the memory.
  if ((outputGridsSeismic_ & IO::RESIDUAL) > 0)
  {
    if (!simbox_->getIsConstantThick())
      multiplyDataByScaleWaveletAndWriteToFile("residuals");
    else {
      for (int l=0 ; l<ntheta_ ; l++) {
        std::string angle     = NRLib::ToString(thetaDeg_[l],1);
        std::string sgriLabel = " Residuals for incidence angle "+angle;
        std::string fileName  = IO::PrefixResiduals() + angle;
        seisData_[l]->setAccessMode(FFTGrid::RANDOMACCESS);
        seisData_[l]->invFFTInPlace();
        seisData_[l]->writeFile(fileName, IO::PathToInversionResults(), simbox_, sgriLabel);
        seisData_[l]->endAccess();
      }
    }
  }

  for(int i = 0 ; i < ntheta_ ; i++)
    delete seisData_[i];
  delete [] seisData_;

  if (modelGeneral_->getVelocityFromInversion()) { //Conversion undefined until prediction ready. Complete it.
    LogKit::WriteHeader("Setup time-to-depth relationship");
    LogKit::LogFormatted(LogKit::Low,"\nUsing Vp velocity field from inversion to map between time and depth grids.\n");
    postAlpha_->setAccessMode(FFTGrid::RANDOMACCESS);
    postAlpha_->expTransf();
    GridMapping       * tdMap      = modelGeneral_->getTimeDepthMapping();
    const GridMapping * dcMap      = modelGeneral_->getTimeCutMapping();
    const Simbox      * timeSimbox = simbox_;
    if(dcMap != NULL)
      timeSimbox = dcMap->getSimbox();

    tdMap->setMappingFromVelocity(postAlpha_, timeSimbox);
    postAlpha_->logTransf();
    postAlpha_->endAccess();
  }

  //NBNB Anne Randi: Skaler traser ihht notat fra Hugo
  if (modelAVOdynamic_->getUseLocalNoise()) {
    FFTGrid * postCovAlpha       = seismicParameters.GetCovAlpha();
    FFTGrid * postCovBeta        = seismicParameters.GetCovBeta();
    FFTGrid * postCovRho         = seismicParameters.GetCovRho();

    seismicParameters.invFFTCovGrids();
    seismicParameters.updatePriorVar();

    postVar0_       = seismicParameters.getPriorVar0(); //Updated variables
    postCovAlpha00_ = seismicParameters.createPostCov00(postCovAlpha);
    postCovBeta00_  = seismicParameters.createPostCov00(postCovBeta);
    postCovRho00_   = seismicParameters.createPostCov00(postCovRho);

    seismicParameters.FFTCovGrids();

    correctAlphaBetaRho(modelSettings_);
  }

  if (!doing4DInversion_) {
    if (writePrediction_)
      ParameterOutput::writeParameters(simbox_, modelGeneral_, modelSettings_, postAlpha_, postBeta_, postRho_,
                                       outputGridsElastic_, fileGrid_, -1, false);

    writeBWPredicted();
  }


    time(&timeend);
    LogKit::LogFormatted(LogKit::DebugLow,"\nTime elapsed :  %d\n",timeend-timestart);

    if(modelSettings->getNumberOfSimulations() > 0)
      simulate(seismicParameters, modelGeneral->getRandomGen());

    seismicParameters.invFFTCovGrids();
    seismicParameters.updatePriorVar();

    if (!modelAVOdynamic->getUseLocalNoise()) {// Already done in crava.cpp if local noise
      postVar0_       = seismicParameters.getPriorVar0(); //Updated variables
      postCovAlpha00_ = seismicParameters.createPostCov00(seismicParameters.GetCovAlpha());
      postCovBeta00_  = seismicParameters.createPostCov00(seismicParameters.GetCovBeta());
      postCovRho00_   = seismicParameters.createPostCov00(seismicParameters.GetCovRho());
    }
    seismicParameters.printPostVariances(postVar0_);

    if((modelSettings->getOutputGridsOther() & IO::CORRELATION) > 0){
      seismicParameters.writeFilePostVariances(postVar0_, postCovAlpha00_, postCovBeta00_, postCovRho00_);
      seismicParameters.writeFilePostCovGrids(modelGeneral->getTimeSimbox());
    }

    int activeAngles = 0; //How many dimensions for local noise interpolation? Turn off for now.
    if(modelAVOdynamic->getUseLocalNoise()==true)
      activeAngles = modelAVOdynamic->getNumberOfAngles();
    if(spatwellfilter != NULL && modelSettings->getFaciesProbFromRockPhysics() == false)
      spatwellfilter->doFiltering(modelGeneral->getWells(),
                                  modelSettings->getNumberOfWells(),
                                  modelSettings->getNoVsFaciesProb(),
                                  activeAngles,
                                  this,
                                  modelAVOdynamic->getLocalNoiseScales(),
                                  seismicParameters);
    if (modelSettings->getEstimateFaciesProb()) {
      bool useFilter = modelSettings->getUseFilterForFaciesProb();
      computeFaciesProb(spatwellfilter, useFilter, seismicParameters);
    }
    if(modelSettings->getKrigingParameter() > 0)
      doPredictionKriging(seismicParameters);

    if(modelSettings->getGenerateSeismicAfterInv())
      computeSyntSeismic(postAlpha_,postBeta_,postRho_);
    //
    // Temporary placement.
    //
    if((modelSettings->getWellOutputFlag() & IO::BLOCKED_WELLS) > 0) {
      modelAVOstatic->writeBlockedWells(modelGeneral->getWells(),modelSettings, modelGeneral->getFaciesNames(), modelGeneral->getFaciesLabel());
    }
    if((modelSettings->getWellOutputFlag() & IO::BLOCKED_LOGS) > 0) {
      LogKit::LogFormatted(LogKit::Low,"\nWARNING: Writing of BLOCKED_LOGS is not implemented yet.\n");
    }
  }
  else{
    LogKit::LogFormatted(LogKit::Low,"\n               ... model built\n");

    computeSyntSeismic(postAlpha_,postBeta_,postRho_);
  }

  postAlpha_->fftInPlace();
  postBeta_->fftInPlace();
  postRho_->fftInPlace();

  seismicParameters.setBackgroundParameters(postAlpha_, postBeta_, postRho_);

  if(!modelSettings->getForwardModeling())
    seismicParameters.FFTCovGrids();

  delete spatwellfilter;

}

Crava::~Crava()
{
  delete [] thetaDeg_;
  delete [] empSNRatio_;
  delete [] theoSNRatio_;
  delete [] modelVariance_;
  delete [] signalVariance_;
  delete [] errorVariance_;
  delete [] dataVariance_;

//  delete postAlpha_;
//  delete postBeta_;
//  delete postRho_;

  if(fprob_ != NULL)
    delete fprob_;
  if(errCorr_ != NULL)
    delete errCorr_;

  for(int i = 0;i<ntheta_;i++)
    delete[] errThetaCov_[i];
  delete [] errThetaCov_;

  if(sigmamdnew_!=NULL)
  {
    for(int i=0;i<nx_;i++)
    {
      for(int j=0;j<ny_;j++)
      {
        if((*sigmamdnew_)(i,j)!=NULL)
        {
          for(int ii=0;ii<3;ii++)
            delete [] (*sigmamdnew_)(i,j)[ii];
          delete [] (*sigmamdnew_)(i,j);
        }
      }
    }
     delete sigmamdnew_;

  }
}

void
Crava::computeDataVariance(void)
{
  //
  // Compute variation in raw seismic
  //
  int rnxp = 2*(nxp_/2+1);
  for(int l=0 ; l < ntheta_ ; l++)
  {
    double  totvar = 0;
    long int ndata = 0;
    dataVariance_[l]=0.0;
    seisData_[l]->setAccessMode(FFTGrid::READ);
    for(int k=0 ; k<nzp_ ; k++)
    {
      double tmpvar1 = 0;
      for(int j=0;j<nyp_;j++)
      {
        double tmpvar2 = 0;
        for(int i=0; i <rnxp; i++)
        {
          float tmp=seisData_[l]->getNextReal();
          if(k < nz_ && j < ny_ &&  i < nx_ && tmp != 0.0)
          {
            tmpvar2 += double(tmp*tmp);
            ndata++;
          }
        }
        tmpvar1 += tmpvar2;
      }
      totvar += tmpvar1;
    }
    seisData_[l]->endAccess();
    if (ndata == 0) {
      dataVariance_[l] = 0.0;
      LogKit::LogFormatted(LogKit::Low,"\nWARNING: All seismic data in stack "+NRLib::ToString(l)+" have zero amplitude.\n");
      TaskList::addTask("Check the seismic data for stack"+NRLib::ToString(l)+". All data have zero amplitude.");
    }
    else {
      dataVariance_[l] = static_cast<float>(totvar/static_cast<double>(ndata));
    }
  }
}

void
Crava::setupErrorCorrelation(const std::vector<Grid2D *> & noiseScale)
{
  //
  //  Setup error correlation matrix
  //
  for(int l=0 ; l < ntheta_ ; l++)
  {
    empSNRatio_[l] = modelAVOdynamic_->getSNRatio(l);
    if(modelAVOdynamic_->getUseLocalNoise() == true) {
      double minScale = noiseScale[l]->FindMin(RMISSING);
      errorVariance_[l] = float(dataVariance_[l]*minScale/empSNRatio_[l]);
    }
    else
      errorVariance_[l] = dataVariance_[l]/empSNRatio_[l];

    if (empSNRatio_[l] < 1.1f)
    {
      LogKit::LogFormatted(LogKit::Low,"\nThe empirical signal-to-noise ratio for angle stack %d is %7.1e. Ratios smaller than\n",l+1,empSNRatio_[l]);
      LogKit::LogFormatted(LogKit::Low," 1 are illegal and CRAVA has to stop. CRAVA was for some reason not able to estimate\n");
      LogKit::LogFormatted(LogKit::Low," this ratio reliably, and you must give it as input to the model file\n\n");
      exit(1);
    }
  }

  Vario * angularCorr = modelAVOdynamic_->getAngularCorr();

  for(int i = 0; i < ntheta_; i++)
    for(int j = 0; j < ntheta_; j++)
      {
        float dTheta = modelAVOdynamic_->getAngle(i) - modelAVOdynamic_->getAngle(j);
        errThetaCov_[i][j] = static_cast<float>(sqrt(errorVariance_[i])
                                                *sqrt(errorVariance_[j])
                                                *angularCorr->corr(dTheta,0));
      }

}

void
Crava::computeVariances(fftw_real     * corrT,
                        ModelSettings * modelSettings)
{
  computeDataVariance();

  setupErrorCorrelation(modelAVOdynamic_->getLocalNoiseScales());

  Wavelet1D ** errorSmooth = new Wavelet1D*[ntheta_];
  float      * paramVar    = new float[ntheta_] ;
  float      * WDCorrMVar  = new float[ntheta_];

  for(int i=0 ; i < ntheta_ ; i++)
  {
    Wavelet1D * wavelet1D = seisWavelet_[i]->createWavelet1DForErrorNorm();
    errorSmooth[i] = new Wavelet1D(wavelet1D,Wavelet::FIRSTORDERFORWARDDIFF);
    delete wavelet1D;

    std::string angle    = NRLib::ToString(thetaDeg_[i], 1);
    std::string fileName = IO::PrefixWavelet() + std::string("Diff_") + angle + IO::SuffixGeneralData();
    errorSmooth[i]->printToFile(fileName);
  }

  // Compute variation in parameters
  for(int i=0 ; i < ntheta_ ; i++)
  {
    paramVar[i]=0.0;
    for(int l=0; l<3 ; l++)
      for(int m=0 ; m<3 ; m++)
        paramVar[i] += static_cast<float>(priorVar0_(l,m))*A_[i][l]*A_[i][m];
  }

  // Compute variation in wavelet
  for(int l=0 ; l < ntheta_ ; l++)
  {
    WDCorrMVar[l] = computeWDCorrMVar(errorSmooth[l],corrT);
  }

  // Compute signal and model variance and theoretical signal-to-noise-ratio
  for(int l=0 ; l < ntheta_ ; l++)
  {
    modelVariance_[l]  = WDCorrMVar[l]*paramVar[l];
    signalVariance_[l] = errorVariance_[l] + modelVariance_[l];
  }

  for(int l=0 ; l < ntheta_ ; l++)
  {
    if (modelAVOdynamic_->getMatchEnergies(l))
    {
      LogKit::LogFormatted(LogKit::Low,"Matching syntethic and empirical energies:\n");
      float gain = sqrt((errorVariance_[l]/modelVariance_[l])*(empSNRatio_[l] - 1.0f));
      seisWavelet_[l]->scale(gain);
      if((modelSettings->getWaveletOutputFlag() & IO::GLOBAL_WAVELETS) > 0 ||
          (modelSettings->getEstimationMode() && modelAVOdynamic_->getEstimateWavelet(l)))
      {
        std::string angle    = NRLib::ToString(thetaDeg_[l], 1);
        std::string fileName = IO::PrefixWavelet() + std::string("EnergyMatched_") + angle;
        seisWavelet_[l]->writeWaveletToFile(fileName, 1.0,false); // dt_max = 1.0;
      }
      modelVariance_[l] *= gain*gain;
      signalVariance_[l] = errorVariance_[l] + modelVariance_[l];
    }
    theoSNRatio_[l] = signalVariance_[l]/errorVariance_[l];
  }

  delete [] paramVar ;
  delete [] WDCorrMVar;
  for(int i=0;i<ntheta_;i++)
    delete errorSmooth[i];
  delete [] errorSmooth;
}

void
Crava::computeElasticImpedanceTimeCovariance(fftw_real * eiCovT,
                                             float     * corrT,
                                             float     * A ) const
{
  double eiVar=0.0;
  for(int l=0; l<3 ; l++)
      for(int m=0 ; m<3 ; m++)
        eiVar += A[l]*static_cast<float>(priorVar0_(l,m))*A[m];

  for(int k=0;k<nzp_;k++)
    eiCovT[k]=static_cast<fftw_real>(eiVar*corrT[k]);
}

void
Crava::computeReflectionCoefficientTimeCovariance(fftw_real * refCovT,
                                                  float     * corrT,
                                                  float     * A ) const
{
  computeElasticImpedanceTimeCovariance(refCovT, corrT, A);

  fftw_real first = refCovT[0];
  fftw_real prev  = refCovT[nzp_-1];

  for(int i=0;i<nzp_-1;i++)
  {
    fftw_real curr = refCovT[i];
    fftw_real next = refCovT[i+1];
    refCovT[i] = 2*curr-next-prev;
    prev =curr;
  }
  refCovT[nzp_-1] = 2*refCovT[nzp_-1] -first-prev;

}


int
Crava::checkScale(void)
{
  std::string scaleWarning1;
  std::string scaleWarning2;
  std::string scaleWarning3;
  std::string scaleWarning4;

  scaleWarning1 = "The observed variability in seismic data is much larger than in the model";
  scaleWarning2 = "The observed variability in seismic data is much less than in the model";
  scaleWarning3 = "Small signal to noise ratio detected";
  scaleWarning4 = "Large signal to noise ratio detected";

  bool thisThetaIsOk;
  int  isOk=0;

  for(int l=0 ; l < ntheta_ ; l++)
  {
    thisThetaIsOk=true;

    if(( dataVariance_[l] > 4 * signalVariance_[l]) && thisThetaIsOk) //1 var 4
    {
      thisThetaIsOk=false;
      if(isOk==0)
      {
        isOk = 1;
        scaleWarningText_ = "Model inconsistency in angle "+NRLib::ToString(int(thetaDeg_[l]+0.5))+" for seismic data:\n"+scaleWarning1+"\n";
      }
      else
      {
        isOk = 1;
        scaleWarningText_ += "Model inconsistency in angle "+NRLib::ToString(int(thetaDeg_[l]+0.5))+" for seismic data:\n"+scaleWarning1+"\n";
      }
    }
    if( (dataVariance_[l] < 0.1 * signalVariance_[l]) && thisThetaIsOk) //1 var 0.1
    {
      thisThetaIsOk=false;
      if(isOk==0)
      {
        isOk = 2;
        scaleWarningText_ = "Model inconsistency in angle "+NRLib::ToString(int(thetaDeg_[l]+0.5))+" for seismic data:\n"+scaleWarning2+"\n";
      }
      else
      {
        isOk = 2;
        scaleWarningText_ += "Model inconsistency in angle "+NRLib::ToString(int(thetaDeg_[l]+0.5))+" for seismic data:\n"+scaleWarning2+"\n";
      }
    }
    if( (modelVariance_[l] < 0.02 * errorVariance_[l]) && thisThetaIsOk)
    {
      thisThetaIsOk=false;
      if(isOk==0)
        {
        isOk = 3;
        scaleWarningText_ = scaleWarning3+" for angle "+NRLib::ToString(int(thetaDeg_[l]+0.5))+"\n";
      }
      else
        {
        isOk = 3;
        scaleWarningText_ += scaleWarning3+" for angle "+NRLib::ToString(int(thetaDeg_[l]+0.5))+"\n";
      }
    }
    if( (modelVariance_[l] > 50.0 * errorVariance_[l]) && thisThetaIsOk)
    {
      thisThetaIsOk=false;
      if(isOk==0)
        {
        isOk = 4;
        scaleWarningText_ = scaleWarning4+" for angle "+NRLib::ToString(int(thetaDeg_[l]+0.5))+"\n";
      }
      else
      {
        isOk = 4;
        scaleWarningText_ += scaleWarning4+" for angle "+NRLib::ToString(int(thetaDeg_[l]+0.5))+"\n";
      }
    }
  }
  return isOk;
}

void
Crava::divideDataByScaleWavelet(const SeismicParametersHolder & seismicParameters)
{
  int i,j,k,l,flag;

  fftw_real*    rData;
  fftw_real     tmp;
  fftw_complex* cData ;
  fftw_complex* adjustmentFactor;


  rfftwnd_plan plan1,plan2;

  rData  = static_cast<fftw_real*>(fftw_malloc(2*(nzp_/2+1)*sizeof(fftw_real)));
  cData  = reinterpret_cast<fftw_complex*>(rData);
  adjustmentFactor= static_cast<fftw_complex*>(fftw_malloc(2*(nzp_/2+1)*sizeof(fftw_real)));

  Wavelet1D* localWavelet ;

  flag   = FFTW_ESTIMATE | FFTW_IN_PLACE;
  plan1  = rfftwnd_create_plan(1,&nzp_,FFTW_REAL_TO_COMPLEX,flag);
  plan2  = rfftwnd_create_plan(1,&nzp_,FFTW_COMPLEX_TO_REAL,flag);

  for(l=0 ; l< ntheta_ ; l++ )
  {
    int dim=seisWavelet_[l]->getDim();
    std::string angle = NRLib::ToString(thetaDeg_[l], 1);
    if(ModelSettings::getDebugLevel() > 0) {
      std::string fileName = IO::PrefixOriginalSeismicData() + "With_Padding_" + angle;
      seisData_[l]->writeStormFile(fileName, simbox_, false, true, true);
    }

    seisData_[l]->setAccessMode(FFTGrid::RANDOMACCESS);
    for(i=0; i < nxp_; i++)
      for(j=0; j< nyp_; j++)
      {
        // gets data
        int iInd=i;
        int jInd=j;

        if(iInd > 3*nx_-1  ){
          iInd = 0;
        }
        if(jInd > 3*ny_-1  ){
          jInd = 0;
        }

        if((iInd > (nxp_+nx_)/2))
          iInd = nxp_-iInd;
        if(iInd >= nx_ )
          iInd = 2*nx_-iInd-1;

        if(jInd > (nyp_+ny_)/2)
          jInd = nyp_-jInd;
        if(jInd >= ny_ )
          jInd = 2*ny_-jInd-1;

        for(k=0;k<nzp_;k++)
        {
          rData[k] = seisData_[l]->getRealValue(i,j,k, true)/static_cast<float>(sqrt(static_cast<float>(nzp_)));

          if(k > nz_)
          {
            float dist = seisData_[l]->getDistToBoundary( k, nz_, nzp_);
            rData[k] *= std::max<float>(1-dist*dist,0);
          }
        }
        rfftwnd_one_real_to_complex(plan1,rData ,cData); // fourier transform of data in profile (i,j)
        // end get data

        // Wavelet local properties
        localWavelet = seisWavelet_[l]->createLocalWavelet1D(iInd,jInd);  //
        double sfLoc =(simbox_->getRelThick(i,j)*seisWavelet_[l]->getLocalStretch(iInd,jInd));// scale factor from thickness stretch + (local stretch when 3D wavelet)

        double relT   = simbox_->getRelThick(i,j);
        double deltaF = static_cast<double>(nz_)*1000.0/(relT*simbox_->getlz()*static_cast<double>(nzp_));
        if(dim==1)
          computeAdjustmentFactor( adjustmentFactor, localWavelet , sfLoc, seisWavelet_[l], seismicParameters, A_[l],static_cast<float>(errThetaCov_[l][l]));
        else
          computeAdjustmentFactor( adjustmentFactor, localWavelet , sfLoc, seisWavelet_[l]->getGlobalWavelet(), seismicParameters, A_[l],static_cast<float>(errThetaCov_[l][l]));

        delete localWavelet;

        for(k=0;k < (nzp_/2 +1);k++) // all complex values
        {
          if( (deltaF*k < highCut_ ) && (deltaF*k > lowCut_ )) //NBNB frequency cleaning
          {
            tmp           = cData[k].re * adjustmentFactor[k].re - cData[k].im * adjustmentFactor[k].im;
            cData[k].im   = cData[k].im * adjustmentFactor[k].re + cData[k].re * adjustmentFactor[k].im;
            cData[k].re   = tmp;
          }
          else
          {
            cData[k].im = 0.0f;
            cData[k].re = 0.0f;
          }
        }
        rfftwnd_one_complex_to_real(plan2 ,cData ,rData);
        for(k=0;k<nzp_;k++)
        {
          seisData_[l]->setRealValue(i,j,k,rData[k]/static_cast<float>(sqrt(static_cast<float>(nzp_))),true);
        }
      }

      if(ModelSettings::getDebugLevel() > 0)
      {
        std::string fileName1 = IO::PrefixReflectionCoefficients() + angle;
        std::string fileName2 = IO::PrefixReflectionCoefficients() + "With_Padding_" + angle;
        std::string sgriLabel = "Reflection coefficients for incidence angle " + angle;
        seisData_[l]->writeFile(fileName1, IO::PathToDebug(), simbox_, sgriLabel);
        seisData_[l]->writeStormFile(fileName2, simbox_, false, true, true);
      }

      LogKit::LogFormatted(LogKit::Medium,"\nInterpolating reflections for angle stack "+angle+": ");
      seisData_[l]->interpolateSeismic(energyTreshold_);

      if(ModelSettings::getDebugLevel() > 0)
      {
        std::string sgriLabel = "Interpolated reflections for incidence angle "+angle;
        std::string fileName1 = IO::PrefixReflectionCoefficients() + "Interpolated_" + angle;
        std::string fileName2 = IO::PrefixReflectionCoefficients()  + "Interpolated_With_Padding_" + angle;
        seisData_[l]->writeFile(fileName1, IO::PathToDebug(), simbox_, sgriLabel);
        seisData_[l]->writeStormFile(fileName2, simbox_, false, true, true);
      }
      seisData_[l]->endAccess();
  }

  fftw_free(rData);
  fftw_free(adjustmentFactor);
  fftwnd_destroy_plan(plan1);
  fftwnd_destroy_plan(plan2);
}


void
Crava::computeAdjustmentFactor(fftw_complex                  * adjustmentFactor,
                               Wavelet1D                     * wLocal,
                               double                          sf,
                               Wavelet                       * wGlobal,
                               const SeismicParametersHolder & seismicParameters,
                               float                         * A,
                               float                           errorVar)
{
// Computes the 1D inversion (of a single cube) with the local wavelet
// and then multiply up with the values of the global wavelet
// in order to adjust the data that inversion is ok with new data.
  float tolFac= 0.05f;

    // computes the time covariance for reflection coefficients rcCovT can be globaly stored
  fftw_real* rcCovT;
  int flag   = FFTW_ESTIMATE | FFTW_IN_PLACE;
  rfftwnd_plan plan1  = rfftwnd_create_plan(1, &nzp_ ,FFTW_REAL_TO_COMPLEX,flag);
  rcCovT = static_cast<fftw_real*>(fftw_malloc(2*(nzp_/2+1)*sizeof(fftw_real)));
  fftw_complex * rcSpecIntens = reinterpret_cast<fftw_complex*>(rcCovT);

  float * corrT = seismicParameters.getPriorCorrTFiltered(nz_, nzp_);
  computeReflectionCoefficientTimeCovariance(rcCovT, corrT, A);
  rfftwnd_one_real_to_complex(plan1,rcCovT ,rcSpecIntens); // operator FFT (not isometric)
  delete [] corrT;

  // computes the time Covariance in the errorterm with wavelet Local can be more efficiently computed
  Wavelet1D *errorSmooth  = new Wavelet1D(wLocal ,Wavelet::FIRSTORDERFORWARDDIFF);
  Wavelet1D *errorSmooth2 = new Wavelet1D(errorSmooth, Wavelet::FIRSTORDERBACKWARDDIFF);
  Wavelet1D *errorSmooth3 = new Wavelet1D(errorSmooth2,Wavelet::FIRSTORDERCENTRALDIFF);
  errorSmooth->fft1DInPlace();
  float normF1 = errorSmooth->findNormWithinFrequencyBand(lowCut_,highCut_);
  errorSmooth->multiplyRAmpByConstant(1.0f/normF1);
  errorSmooth3->fft1DInPlace();
  float normF3 = errorSmooth3->findNormWithinFrequencyBand(lowCut_,highCut_);
  errorSmooth3->multiplyRAmpByConstant(1.0f/normF3);
  wLocal->fft1DInPlace();

  // Wavelet global properties sets order of size
  wGlobal->fft1DInPlace();
  float modW = wGlobal->getNorm();// note the wavelet norm is in time domain. In frequency domain we have an additional factor float(nzp_);
                                  // this is because we define the wavelet as an operator hence the fft is not norm preserving.
  modW *= modW;
  float maxfrequency = static_cast<float>((nzp_/2)*1000.0*nz_)/static_cast<float>(simbox_->getlz()*nzp_);
  modW *= maxfrequency/static_cast<float>(highCut_); // this is the mean squared sum over relevant frequency band.(up to highCut_)
                                                     // Makes the problem less sensitive to the padding size

  for(int k=0;k< (nzp_/2 +1);k++)
  {
    fftw_complex wLoc   = wLocal->getCAmp(k,static_cast<float>(sf)); // returns complex conjugate ...see below
    float wLoc2          = (wLoc.re*wLoc.re+wLoc.im*wLoc.im);
    fftw_complex wGlob  = wGlobal->getCAmp(k);
    float wGlob2          = (wGlob.re*wGlob.re+wGlob.im*wGlob.im);

    fftw_complex eSkLoc = wLocal->getCAmp(k,static_cast<float>(sf));
    //fftw_complex eSkLoc = errorSmooth->getCAmp(k,static_cast<float>(sf));
    float sizeESkLoc    = (eSkLoc.re*eSkLoc.re  +  eSkLoc.im*eSkLoc.im)/(wLocal->getNorm()*wLocal->getNorm());
    eSkLoc              = errorSmooth3->getCAmp(k,static_cast<float>(sf));
    sizeESkLoc         += eSkLoc.re*eSkLoc.re  +  eSkLoc.im*eSkLoc.im;

    fftw_complex eSkGlob = wLocal->getCAmp(k);
    //fftw_complex eSkLoc = errorSmooth->getCAmp(k);
    float sizeESkGlob    = (eSkGlob.re*eSkGlob.re  +  eSkGlob.im*eSkGlob.im)/(wLocal->getNorm()*wLocal->getNorm());;
    eSkGlob              = errorSmooth3->getCAmp(k);
    sizeESkGlob         += eSkGlob.re*eSkGlob.re  +  eSkGlob.im*eSkGlob.im;

    float sigma2Loc  = errorVar*(wnc_ +(1.0f-wnc_)*0.5f*sizeESkLoc);
    float sigma2Glob = errorVar*(wnc_ +(1.0f-wnc_)*0.5f*sizeESkGlob);
    float tau2   = fabs(static_cast<float>(rcSpecIntens[k].re));

    float wGlob2Adj = wGlob2;
    float wLoc2Adj = wLoc2;

    if(wGlob2 <  tolFac * modW)
      wGlob2Adj =   float(0.5f*(wGlob2 + tolFac * modW));
    if(wLoc2 <  tolFac * modW)
      wLoc2Adj =   float(0.5f*(wLoc2 + tolFac * modW));
    if(wLoc2Adj <  tolFac * wGlob2Adj)
    {
      wLoc2Adj =   float(0.5f*(wLoc2 + tolFac * wGlob2Adj));
      sigma2Loc *= wLoc2Adj/wLoc2;
    }

    adjustmentFactor[k].re = (wLoc.re/(tau2*wLoc2Adj+sigma2Loc )) / ( wGlob2Adj/(tau2*wGlob2+sigma2Glob)) ;
    adjustmentFactor[k].im = (wLoc.im/(tau2*wLoc2Adj+sigma2Loc )) / ( wGlob2Adj/(tau2*wGlob2+sigma2Glob)) ; // this is where we use thatthe value we have is the complex conjugate (otherwise it should be minus)
  }
  // clean up
  delete errorSmooth;
  delete errorSmooth2;
  delete errorSmooth3;
  fftwnd_destroy_plan(plan1);
  fftw_free(rcCovT);
}

void
Crava::multiplyDataByScaleWaveletAndWriteToFile(const std::string & typeName)
{
  int i,j,k,l,flag;

  fftw_real*    rData;
  fftw_real     tmp;
  fftw_complex* cData;
  fftw_complex scaleWVal;
  rfftwnd_plan plan1,plan2;

  rData  = static_cast<fftw_real*>(fftw_malloc(2*(nzp_/2+1)*sizeof(fftw_real)));
  cData  = reinterpret_cast<fftw_complex*>(rData);

  flag   = FFTW_ESTIMATE | FFTW_IN_PLACE;
  plan1  = rfftwnd_create_plan(1, &nzp_ ,FFTW_REAL_TO_COMPLEX,flag);
  plan2  = rfftwnd_create_plan(1,&nzp_,FFTW_COMPLEX_TO_REAL,flag);

  Wavelet1D* localWavelet;

  for(l=0 ; l< ntheta_ ; l++ )
  {
    seisData_[l]->setAccessMode(FFTGrid::RANDOMACCESS);
    seisData_[l]->invFFTInPlace();

    for(i=0; i < nx_; i++)
      for(j=0; j< ny_; j++)
      {
        float sf = static_cast<float>(simbox_->getRelThick(i,j))*seisWavelet_[l]->getLocalStretch(i,j);

        for(k=0;k<nzp_;k++)
        {
          rData[k] = seisData_[l]->getRealValue(i,j,k, true)/static_cast<float>(sqrt(static_cast<float>(nzp_)));
        }

        rfftwnd_one_real_to_complex(plan1,rData ,cData);
        localWavelet = seisWavelet_[l]->createLocalWavelet1D(i,j);

        for(k=0;k < (nzp_/2 +1);k++) // all complex values
        {
            scaleWVal    =  localWavelet->getCAmp(k,sf);    // NBNB change here
          //scaleWVal    =  localWavelet->getCAmp(k);
          // note scaleWVal is acctually the value of the complex conjugate
          // (see definition of getCAmp)
          tmp           = cData[k].re * scaleWVal.re + cData[k].im * scaleWVal.im;
          cData[k].im   = cData[k].im * scaleWVal.re - cData[k].re * scaleWVal.im;
          cData[k].re   = tmp;
        }
        delete localWavelet;
        rfftwnd_one_complex_to_real(plan2 ,cData ,rData);
        for(k=0;k<nzp_;k++)
        {
          seisData_[l]->setRealValue(i,j,k,rData[k]/static_cast<float>(sqrt(static_cast<double>(nzp_))),true);
        }

      }
      std::string angle     = NRLib::ToString(thetaDeg_[l],1);
      std::string sgriLabel = typeName + " for incidence angle "+angle;
      std::string fileName  = typeName + "_" + angle;
      seisData_[l]->writeFile(fileName, IO::PathToInversionResults(), simbox_, sgriLabel);
      seisData_[l]->endAccess();
  }

  fftw_free(rData);
  fftwnd_destroy_plan(plan1);
  fftwnd_destroy_plan(plan2);
}


int
Crava::computePostMeanResidAndFFTCov(ModelGeneral            *  modelGeneral,
                                     const Simbox            *  simbox,
                                     SeismicParametersHolder &  seismicParameters,
                                     FFTGrid                 *& postAlpha,
                                     FFTGrid                 *& postBeta,
                                     FFTGrid                 *& postRho,
                                     FFTGrid                 *& errCorr,
                                     FFTGrid                 ** seisData,
                                     Wavelet                 ** seisWavelet,
                                     double                  ** errThetaCov,
                                     float                   ** A,
                                     float                   *  thetaDeg,
                                     int                        ntheta,
                                     float                      lowCut,
                                     float                      highCut,
                                     float                      wnc,
                                     int                        nz,
                                     int                        nzp,
                                     int                        nyp,
                                     int                        nxp,
                                     int                        n_threads,
                                     bool                       fileGrid)
{
  LogKit::WriteHeader("Posterior model / Performing Inversion");

  double wall=0.0, cpu=0.0;
  TimeKit::getTime(wall,cpu);

  Wavelet1D * diff1Operator = new Wavelet1D(Wavelet::FIRSTORDERFORWARDDIFF,nz,nzp);
  Wavelet1D * diff2Operator = new Wavelet1D(diff1Operator,Wavelet::FIRSTORDERBACKWARDDIFF);
  Wavelet1D * diff3Operator = new Wavelet1D(diff2Operator,Wavelet::FIRSTORDERCENTRALDIFF);

  diff1Operator->fft1DInPlace();
  delete diff2Operator;
  diff3Operator->fft1DInPlace();

  Wavelet1D ** errorSmooth  = new Wavelet1D * [ntheta];
  Wavelet1D ** errorSmooth3 = new Wavelet1D * [ntheta];

  makeErrorSmooth(errorSmooth,
                  errorSmooth3,
                  seisData,
                  seisWavelet,
                  thetaDeg,
                  ntheta);

  FFTGrid * postCovAlpha       = seismicParameters.GetCovAlpha();
  FFTGrid * postCovBeta        = seismicParameters.GetCovBeta();
  FFTGrid * postCovRho         = seismicParameters.GetCovRho();
  FFTGrid * postCrCovAlphaBeta = seismicParameters.GetCrCovAlphaBeta();
  FFTGrid * postCrCovAlphaRho  = seismicParameters.GetCrCovAlphaRho();
  FFTGrid * postCrCovBetaRho   = seismicParameters.GetCrCovBetaRho();

  if (modelGeneral->getIs4DActive()) {
    std::vector<FFTGrid *> sigma(6);
    sigma[0] = postCovAlpha;
    sigma[1] = postCrCovAlphaBeta;
    sigma[2] = postCrCovAlphaRho;
    sigma[3] = postCovBeta;
    sigma[4] = postCrCovBetaRho;
    sigma[5] = postCovRho;
    modelGeneral->mergeCovariance(sigma); //To avoid a second FFT of these.
  }
  else
    seismicParameters.FFTCovGrids();

  Wavelet1D ** seisWaveletForNorm = new Wavelet1D * [ntheta];

  for (int l = 0 ; l < ntheta ; l++) {
    seisWaveletForNorm[l] = seisWavelet[l]->createWavelet1DForErrorNorm();
    seisWaveletForNorm[l]->fft1DInPlace();
    if (simbox->getIsConstantThick()) {
      seisWavelet[l]->fft1DInPlace();
    }
  }

  postCovAlpha      ->setAccessMode(FFTGrid::READ);
  postCovBeta       ->setAccessMode(FFTGrid::READ);
  postCovRho        ->setAccessMode(FFTGrid::READ);
  postCrCovAlphaBeta->setAccessMode(FFTGrid::READ);
  postCrCovAlphaRho ->setAccessMode(FFTGrid::READ);
  postCrCovBetaRho  ->setAccessMode(FFTGrid::READ);

  postAlpha->setAccessMode(FFTGrid::READANDWRITE);
  postBeta ->setAccessMode(FFTGrid::READANDWRITE);
  postRho  ->setAccessMode(FFTGrid::READANDWRITE);

  errCorr->fftInPlace();
  errCorr->setAccessMode(FFTGrid::READ);

  // Computes the posterior mean first below the covariance is computed
  // To avoid to many grids in memory at the same time

  LogKit::LogFormatted(LogKit::Low,"\nBuilding posterior distribution:");
  float monitorSize = std::max(1.0f, static_cast<float>(nzp)*0.02f);
  float nextMonitor = monitorSize;
  std::cout
    << "\n  0%       20%       40%       60%       80%      100%"
    << "\n  |    |    |    |    |    |    |    |    |    |    |  "
    << "\n  ^";

#ifdef PARALLEL
  if (fileGrid) {
    n_threads = 1; // Parallelization doesn't work when using FFTFileGrid due to disk access.
  }
  int  chunk_size = 1;
#pragma omp parallel for schedule(dynamic, chunk_size) num_threads(n_threads)
#endif

  for (int k = 0 ; k < nzp ; k++) { // We are now in frequency (Fourier) domain

    fftw_complex *  errMult1    = new fftw_complex[ntheta];
    fftw_complex *  errMult2    = new fftw_complex[ntheta];
    fftw_complex *  errMult3    = new fftw_complex[ntheta];

    fftw_complex *  ijkData     = new fftw_complex[ntheta];
    fftw_complex *  ijkDataMean = new fftw_complex[ntheta];
    fftw_complex *  ijkRes      = new fftw_complex[ntheta];
    fftw_complex *  ijkMean     = new fftw_complex[3];
    fftw_complex *  ijkAns      = new fftw_complex[3];

    fftw_complex ** K           = allocateFFTComplex(ntheta, 3     );
    fftw_complex ** KS          = allocateFFTComplex(ntheta, 3     );
    fftw_complex ** KScc        = allocateFFTComplex(3     , ntheta);
    fftw_complex ** parVar      = allocateFFTComplex(3     , 3     );
    fftw_complex ** reduceVar   = allocateFFTComplex(3     , 3     );
    fftw_complex ** margVar     = allocateFFTComplex(ntheta, ntheta);
    fftw_complex ** errVar      = allocateFFTComplex(ntheta, ntheta);

    float realFrequency         = static_cast<float>((nz*1000.0f)/(simbox->getlz()*nzp)*std::min(k,nzp - k)); // the physical frequency
    bool  invert_frequency      = realFrequency > lowCut*simbox->getMinRelThick() && realFrequency < highCut;

    if (invert_frequency) {
      fillErrMultVectors(errMult1,
                         errMult2,
                         errMult3,
                         K,
                         diff1Operator->getCAmp(k),     // defines content of kD
                         diff3Operator->getCAmp(k),     // defines  kD3
                         seisWavelet,
                         seisWaveletForNorm,
                         errorSmooth3,
                         A,
                         simbox->getIsConstantThick(),
                         ntheta,
                         k);
    }

    int cnxp = nxp/2+1;

    for (int j = 0 ; j < nyp ; j++) {
      for (int i = 0 ; i < cnxp ; i++) {

        fftw_complex ijkErrCorr;

        if (fileGrid) {
          ijkMean[0] = postAlpha->getNextComplex();
          ijkMean[1] = postBeta ->getNextComplex();
          ijkMean[2] = postRho  ->getNextComplex();

          for (int l = 0 ; l < ntheta ; l++) {
            ijkData[l] = seisData[l]->getNextComplex();
            ijkRes[l]  = ijkData[l];
          }
          ijkErrCorr = errCorr->getNextComplex();

          seismicParameters.getNextParameterCovariance(parVar, modelGeneral->getIs4DActive());
        }
        else {
          ijkMean[0] = postAlpha->getComplexValue(i, j, k, true);
          ijkMean[1] = postBeta ->getComplexValue(i, j, k, true);
          ijkMean[2] = postRho  ->getComplexValue(i, j, k, true);

          for (int l = 0 ; l < ntheta ; l++) {
            ijkData[l] = seisData[l]->getComplexValue(i, j, k, true);
            ijkRes[l]  = ijkData[l];
          }
          ijkErrCorr = errCorr->getComplexValue(i, j, k, true);

          seismicParameters.getParameterCovariance(parVar, i, j, k, modelGeneral->getIs4DActive());
        }

        if (invert_frequency) {
          invertSeismicData(parVar,
                            ijkMean,
                            ijkData,
                            ijkDataMean,
                            ijkRes,
                            ijkAns,
                            ijkErrCorr,
                            errMult1,
                            errMult2,
                            errMult3,
                            K,
                            KS,
                            KScc,
                            margVar,
                            reduceVar,
                            errVar,
                            errThetaCov,
                            ntheta,
                            wnc);
        }

        if (fileGrid) {
          for (int l=0 ; l<ntheta ; l++) {
            seisData[l]     ->setNextComplex(ijkRes[l]);
          }
          postAlpha         ->setNextComplex(ijkMean[0]);
          postBeta          ->setNextComplex(ijkMean[1]);
          postRho           ->setNextComplex(ijkMean[2]);
          postCovAlpha      ->setNextComplex(parVar[0][0]);
          postCovBeta       ->setNextComplex(parVar[1][1]);
          postCovRho        ->setNextComplex(parVar[2][2]);
          postCrCovAlphaBeta->setNextComplex(parVar[0][1]);
          postCrCovAlphaRho ->setNextComplex(parVar[0][2]);
          postCrCovBetaRho  ->setNextComplex(parVar[1][2]);
        }
        else {
          for (int l=0 ; l<ntheta ; l++) {
            seisData[l]     ->setComplexValue(i, j, k, ijkRes[l]   , true);
          }
          postAlpha         ->setComplexValue(i, j, k, ijkMean[0]  , true);
          postBeta          ->setComplexValue(i, j, k, ijkMean[1]  , true);
          postRho           ->setComplexValue(i, j, k, ijkMean[2]  , true);
          postCovAlpha      ->setComplexValue(i, j, k, parVar[0][0], true);
          postCovBeta       ->setComplexValue(i, j, k, parVar[1][1], true);
          postCovRho        ->setComplexValue(i, j, k, parVar[2][2], true);
          postCrCovAlphaBeta->setComplexValue(i, j, k, parVar[0][1], true);
          postCrCovAlphaRho ->setComplexValue(i, j, k, parVar[0][2], true);
          postCrCovBetaRho  ->setComplexValue(i, j, k, parVar[1][2], true);
        }
      }
    }

    delete [] errMult1;
    delete [] errMult2;
    delete [] errMult3;

    delete [] ijkData;
    delete [] ijkDataMean;
    delete [] ijkRes;
    delete [] ijkMean ;
    delete [] ijkAns;

    for (int i = 0; i < 3; i++) {
      delete [] KScc[i];
      delete [] parVar[i] ;
      delete [] reduceVar[i];
    }
    delete [] KScc;
    delete [] parVar;
    delete [] reduceVar;

    for(int i = 0; i < ntheta; i++) {
      delete [] K[i];
      delete [] KS[i];
      delete [] margVar[i] ;
      delete [] errVar[i] ;
    }
    delete [] K;
    delete [] KS;
    delete [] margVar;
    delete [] errVar;

    // Log progress
    if (k+1 >= static_cast<int>(nextMonitor))
    {
      nextMonitor += monitorSize;
      std::cout << "^";
      fflush(stdout);
    }
  }
  std::cout << "\n";

  postAlpha->endAccess();
  postBeta->endAccess();
  postRho->endAccess();

  postCovAlpha      ->endAccess();
  postCovBeta       ->endAccess();
  postCovRho        ->endAccess();
  postCrCovAlphaBeta->endAccess();
  postCrCovAlphaRho ->endAccess();
  postCrCovBetaRho  ->endAccess();
  errCorr           ->endAccess();

  postAlpha->invFFTInPlace();
  postBeta ->invFFTInPlace();
  postRho  ->invFFTInPlace();

  for (int l=0 ; l<ntheta ; l++)
    seisData[l]->endAccess();

  delete diff1Operator;
  delete diff3Operator;

  for(int i = 0; i < ntheta; i++)
  {
    delete errorSmooth[i];
    delete errorSmooth3[i];
    delete seisWaveletForNorm[i];
  }

  delete [] errorSmooth3;
  delete [] errorSmooth;
  delete [] seisWaveletForNorm;

  Timings::setTimeInversion(wall,cpu);
  return(0);
}

//--------------------------------------------------------------------
void
Crava::makeErrorSmooth(Wavelet1D **& errorSmooth,
                       Wavelet1D **& errorSmooth3,
                       FFTGrid   **  seisData,
                       Wavelet   **  seisWavelet,
                       float     *   thetaDeg,
                       int           ntheta)
{
  Wavelet1D * wavelet1D    = NULL;
  Wavelet1D * errorSmooth2 = NULL;

  for (int l = 0; l < ntheta ; l++)
  {
    seisData_[l]->setAccessMode(FFTGrid::READANDWRITE);

    wavelet1D       = seisWavelet[l]->createWavelet1DForErrorNorm();
    errorSmooth[l]  = new Wavelet1D(wavelet1D     , Wavelet::FIRSTORDERFORWARDDIFF);
    errorSmooth2    = new Wavelet1D(errorSmooth[l], Wavelet::FIRSTORDERBACKWARDDIFF);
    errorSmooth3[l] = new Wavelet1D(errorSmooth2  , Wavelet::FIRSTORDERCENTRALDIFF);

    std::string angle    = NRLib::ToString(thetaDeg[l], 1);
    std::string fileName = std::string("ErrorSmooth_") + angle + IO::SuffixGeneralData();
    errorSmooth3[l]->printToFile(fileName);
    errorSmooth3[l]->fft1DInPlace();

    fileName = IO::PrefixWavelet() + angle + IO::SuffixGeneralData();
    wavelet1D->printToFile(fileName);
    wavelet1D->fft1DInPlace();

    fileName = std::string("FourierWavelet_") + angle + IO::SuffixGeneralData();
    wavelet1D->printToFile(fileName);

    delete errorSmooth2;
    delete wavelet1D;
  }
}

//-------------------------------------------------------------
fftw_complex **
Crava::allocateFFTComplex(int m, int n)
{
  fftw_complex ** C = new fftw_complex * [m];
  for (int i = 0 ; i < m ; i++)
    C[i] = new fftw_complex[n];
  return C;
}

//--------------------------------------------------------------------
void
Crava::fillErrMultVectors(fftw_complex *& errMult1,           // Filled in method
                          fftw_complex *& errMult2,           // Filled in method
                          fftw_complex *& errMult3,           // Filled in method
                          fftw_complex ** K,
                          fftw_complex    kD,
                          fftw_complex    kD3,
                          Wavelet      ** seisWavelet,
                          Wavelet1D    ** seisWaveletForNorm,
                          Wavelet1D    ** errorSmooth3,
                          float        ** A,
                          bool            constThickness,
                          int             n,                  // ntheta
                          int             k)
{
  if (constThickness)
  {
    fftw_complex * kW = new fftw_complex[n];

    // defines content of K=WDA
    fillkW(k, kW, seisWavelet);

    lib_matrProdScalVecCpx(kD, kW, n);
    lib_matrProdDiagCpxR(kW, A, n, 3, K);         // set (WDA) K

    // defines error-term multipliers
    fillkWNorm(k, errMult1, seisWaveletForNorm);  // set (kWNorm) errMult1
    fillkWNorm(k, errMult2, errorSmooth3);        // set (kWD3Norm) errMult2
    lib_matrFillOnesVecCpx(errMult3, n);          // set errMult3
    delete [] kW;
  }
  else {
    // defines content of K = DA
    lib_matrFillValueVecCpx(kD, errMult1, n);      // errMult1 used as dummy
    lib_matrProdDiagCpxR(errMult1, A, n, 3, K);    // set K (= DA)

    // defines error-term multipliers
    lib_matrFillOnesVecCpx(errMult1, n);           // set errMult1
    for (int l=0 ; l < n ; l++) {
      errMult1[l].re /= seisWavelet[l]->getNorm(); // set errMult1
    }

    lib_matrFillValueVecCpx(kD3, errMult2, n);     // set errMult2
    for (int l=0 ; l < n ; l++) {
      float fac = 1.0f/errorSmooth3[l]->getNorm();
      errMult2[l].re  *= fac;                      // set errMult2
      errMult2[l].im  *= fac;                      // set errMult2
    }
    fillInverseAbskWRobust(k,errMult3,seisWaveletForNorm); // set errMult3
  }
}

void
Crava::invertSeismicData(fftw_complex **& parVar,      // Filled in method
                         fftw_complex  *& ijkMean,     // Filled in method
                         fftw_complex   * ijkData,
                         fftw_complex   * ijkDataMean,
                         fftw_complex   * ijkRes,
                         fftw_complex   * ijkAns,
                         fftw_complex     ijkErrCorr,
                         fftw_complex  *& errMult1,
                         fftw_complex  *& errMult2,
                         fftw_complex  *& errMult3,
                         fftw_complex  ** K,
                         fftw_complex  ** KS,
                         fftw_complex  ** KScc,
                         fftw_complex  ** margVar,
                         fftw_complex  ** reduceVar,
                         fftw_complex  ** errVar,
                         double        ** errThetaCov,
                         int              n,           // ntheta
                         float            wnc)
{
  getNextErrorVariance(errVar,
                       errMult1,
                       errMult2,
                       errMult3,
                       ijkErrCorr,
                       errThetaCov,
                       n, wnc);

  lib_matrProdCpx(K, parVar, n, 3 ,3, KS);                   // KS is defined here
  lib_matrProdAdjointCpx(KS, K, n, 3 ,n, margVar);           // margVar = (K)S(K)' is defined here
  lib_matrAddMatCpx(errVar, n, n, margVar);                  // errVar is added to margVar = (WDA)S(WDA)'  + errVar

  int cholFlag = lib_matrCholCpx(n, margVar);                // Choleskey factor of margVar is Defined

  if (cholFlag == 0) {                                       // If not, posterior = prior
    lib_matrAdjoint(KS, n, 3, KScc);                         // WDAScc is adjoint of WDAS
    lib_matrAXeqBMatCpx(n, margVar, KS, 3);                  // redefines WDAS
    lib_matrProdCpx(KScc, KS, 3, n, 3, reduceVar);           // defines reduceVar
    lib_matrSubtMatCpx(reduceVar, 3, 3, parVar);             // redefines parVar as the posterior solution

    lib_matrProdMatVecCpx(K, ijkMean, n, 3, ijkDataMean);    //  defines content of ijkDataMean
    lib_matrSubtVecCpx(ijkDataMean, n, ijkData);             //  redefines content of ijkData

    lib_matrProdAdjointMatVecCpx(KS, ijkData, 3, n, ijkAns); // defines ijkAns

    lib_matrAddVecCpx(ijkAns, 3, ijkMean);                   // redefines ijkMean
    lib_matrProdMatVecCpx(K,ijkMean, n, 3, ijkData);         // redefines ijkData
    lib_matrSubtVecCpx(ijkData, n, ijkRes);                  // redefines ijkRes
  }
}

//--------------------------------------------------------------------
void
Crava::getNextErrorVariance(fftw_complex **& errVar,
                            fftw_complex   * errMult1,
                            fftw_complex   * errMult2,
                            fftw_complex   * errMult3,
                            fftw_complex     ijkErrCorr,
                            double        ** errThetaCov,
                            int              ntheta,
                            float            wnc) const
{
  fftw_complex ijkErrLam;
  ijkErrLam.re = static_cast<float>(sqrt(ijkErrCorr.re * ijkErrCorr.re));
  ijkErrLam.im = 0.0;

  for(int l = 0; l < ntheta; l++ ) {
    for(int m = 0; m < ntheta; m++ ) { // Note we multiply kWNorm[l] and comp.conj(kWNorm[m]) hence the + and not a minus as in pure multiplication
      errVar[l][m].re  = static_cast<float>( 0.5*(1.0-wnc)*errThetaCov[l][m] * ijkErrLam.re * ( errMult1[l].re *  errMult1[m].re +  errMult1[l].im *  errMult1[m].im));
      errVar[l][m].re += static_cast<float>( 0.5*(1.0-wnc)*errThetaCov[l][m] * ijkErrLam.re * ( errMult2[l].re *  errMult2[m].re +  errMult2[l].im *  errMult2[m].im));
      if(l==m) {
        errVar[l][m].re += static_cast<float>( wnc*errThetaCov[l][m] * errMult3[l].re  * errMult3[l].re);
        errVar[l][m].im  = 0.0;
      }
      else {
        errVar[l][m].im  = static_cast<float>( 0.5*(1.0-wnc)*errThetaCov[l][m] * ijkErrLam.re * (-errMult1[l].re * errMult1[m].im + errMult1[l].im * errMult1[m].re));
        errVar[l][m].im += static_cast<float>( 0.5*(1.0-wnc)*errThetaCov[l][m] * ijkErrLam.re * (-errMult2[l].re * errMult2[m].im + errMult2[l].im * errMult2[m].re));
      }
    }
  }
}

void
Crava::fillkW(int k, fftw_complex* kW, Wavelet** seisWavelet)
{
  for(int l = 0; l < ntheta_; l++)
  {
    kW[l].re  =  float( seisWavelet[l]->getCAmp(k).re );
    kW[l].im  = -float( seisWavelet[l]->getCAmp(k).im ); // adjust for complex conjugate in getCAmp(k)
  }
}

void
Crava::fillkWNorm(int k, fftw_complex* kWNorm, Wavelet1D** wavelet )
{
  int l;
  for(l = 0; l < ntheta_; l++)
  {
    kWNorm[l].re   =  float( wavelet[l]->getCAmp(k).re/wavelet[l]->getNorm());
    kWNorm[l].im   = -float( wavelet[l]->getCAmp(k).im/wavelet[l]->getNorm()); // // adjust for complex conjugate in getCAmp(k)
  }
}

void
Crava::fillInverseAbskWRobust(int k, fftw_complex* invkW ,Wavelet1D** seisWaveletForNorm)
{
  int l;
  float modulus,modulusFine,maxMod;
  fftw_complex value;
  fftw_complex valueFine;
  for(l = 0; l < ntheta_; l++)
  {
    value  = seisWaveletForNorm[l]->getCAmp(k);
    valueFine = seisWaveletForNorm[l]->getCAmp(k,0.999f);// denser sampling of wavelet

    modulus      = value.re*value.re + value.im*value.im;
    modulusFine  = valueFine.re*valueFine.re + valueFine.im*valueFine.im;
    maxMod       = std::max(modulus,modulusFine);

    if(maxMod > 0.0)
    {
      invkW[l].re = float( 1.0/sqrt(maxMod) );
      invkW[l].im = 0.0f;
    }
    else
    {
      invkW[l].re  =  seisWaveletForNorm[l]->getNorm()*nzp_*nzp_*100.0f; // a big number
      invkW[l].im  =  0.0; // a big number
    }
  }
}

std::complex<double>
Crava::SetComplexNumber(const fftw_complex & c)
{
  return std::complex<double>(c.re, c.im);
}

void
Crava::SetComplexVector(NRLib::ComplexVector & V,
                        fftw_complex         * v)
{
  for (int l=0; l < ntheta_; l++) {
    V(l) = std::complex<double>(v[l].re, v[l].im);
  }
}


void
Crava::doPredictionKriging(SeismicParametersHolder & seismicParameters)
{
  if(writePrediction_ == true) { //No need to do this if output not requested.
    double wall2=0.0, cpu2=0.0;
    TimeKit::getTime(wall2,cpu2);
    doPostKriging(seismicParameters, *postAlpha_, *postBeta_, *postRho_);
    Timings::setTimeKrigingPred(wall2,cpu2);
    ParameterOutput::writeParameters(simbox_, modelGeneral_, modelSettings_, postAlpha_, postBeta_, postRho_,
                                     outputGridsElastic_, fileGrid_, -1, true);
  }
}

int
Crava::simulate(SeismicParametersHolder & seismicParameters, RandomGen * randomGen)
{
  LogKit::WriteHeader("Simulating from posterior model");

  double wall=0.0, cpu=0.0;
  TimeKit::getTime(wall,cpu);

  if(nSim_>0)
  {
    bool kriging = (krigingParameter_ > 0);
    FFTGrid * postCovAlpha       = seismicParameters.GetCovAlpha();
    FFTGrid * postCovBeta        = seismicParameters.GetCovBeta();
    FFTGrid * postCovRho         = seismicParameters.GetCovRho();
    FFTGrid * postCrCovAlphaBeta = seismicParameters.GetCrCovAlphaBeta();
    FFTGrid * postCrCovAlphaRho  = seismicParameters.GetCrCovAlphaRho();
    FFTGrid * postCrCovBetaRho   = seismicParameters.GetCrCovBetaRho();

    assert( postCovAlpha->getIsTransformed() );
    assert( postCovBeta->getIsTransformed() );
    assert( postCovRho->getIsTransformed() );
    assert( postCrCovAlphaBeta->getIsTransformed() );
    assert( postCrCovAlphaRho->getIsTransformed() );
    assert( postCrCovBetaRho->getIsTransformed() );

    int             simNr,i,j,k,l;
    fftw_complex ** ijkPostCov;
    fftw_complex *  ijkSeed;
    FFTGrid *       seed0;
    FFTGrid *       seed1;
    FFTGrid *       seed2;

    ijkPostCov = new fftw_complex*[3];
    for(l=0;l<3;l++)
      ijkPostCov[l]=new fftw_complex[3];

    ijkSeed = new fftw_complex[3];

    seed0 =  createFFTGrid();
    seed1 =  createFFTGrid();
    seed2 =  createFFTGrid();
    seed0->createComplexGrid();
    seed1->createComplexGrid();
    seed2->createComplexGrid();

    // long int timestart, timeend;

    for(simNr = 0; simNr < nSim_;  simNr++)
    {
      // time(&timestart);

      seed0->fillInComplexNoise(randomGen);
      seed1->fillInComplexNoise(randomGen);
      seed2->fillInComplexNoise(randomGen);

      postCovAlpha      ->setAccessMode(FFTGrid::READ);
      postCovBeta       ->setAccessMode(FFTGrid::READ);
      postCovRho        ->setAccessMode(FFTGrid::READ);
      postCrCovAlphaBeta->setAccessMode(FFTGrid::READ);
      postCrCovAlphaRho ->setAccessMode(FFTGrid::READ);
      postCrCovBetaRho  ->setAccessMode(FFTGrid::READ);
      seed0 ->setAccessMode(FFTGrid::READANDWRITE);
      seed1 ->setAccessMode(FFTGrid::READANDWRITE);
      seed2 ->setAccessMode(FFTGrid::READANDWRITE);

      int cnxp=nxp_/2+1;
      int cholFlag;
      for(k = 0; k < nzp_; k++)
        for(j = 0; j < nyp_; j++)
          for(i = 0; i < cnxp; i++)
          {
            ijkPostCov[0][0] = postCovAlpha      ->getNextComplex();
            ijkPostCov[1][1] = postCovBeta       ->getNextComplex();
            ijkPostCov[2][2] = postCovRho        ->getNextComplex();
            ijkPostCov[0][1] = postCrCovAlphaBeta->getNextComplex();
            ijkPostCov[0][2] = postCrCovAlphaRho ->getNextComplex();
            ijkPostCov[1][2] = postCrCovBetaRho  ->getNextComplex();

            ijkPostCov[1][0].re =  ijkPostCov[0][1].re;
            ijkPostCov[1][0].im = -ijkPostCov[0][1].im;
            ijkPostCov[2][0].re =  ijkPostCov[0][2].re;
            ijkPostCov[2][0].im = -ijkPostCov[0][2].im;
            ijkPostCov[2][1].re =  ijkPostCov[1][2].re;
            ijkPostCov[2][1].im = -ijkPostCov[1][2].im;

            ijkSeed[0]=seed0->getNextComplex();
            ijkSeed[1]=seed1->getNextComplex();
            ijkSeed[2]=seed2->getNextComplex();

            cholFlag = lib_matrCholCpx(3,ijkPostCov);  // Choleskey factor of posterior covariance write over ijkPostCov
            if(cholFlag == 0)
            {
              lib_matrProdCholVec(3,ijkPostCov,ijkSeed); // write over ijkSeed
            }
            else
            {
              for(l=0; l< 3;l++)
              {
                ijkSeed[l].re =0.0;
                ijkSeed[l].im = 0.0;
              }

            }
            seed0->setNextComplex(ijkSeed[0]);
            seed1->setNextComplex(ijkSeed[1]);
            seed2->setNextComplex(ijkSeed[2]);
          }

          postCovAlpha->endAccess();  //
          postCovBeta->endAccess();   //
          postCovRho->endAccess();
          postCrCovAlphaBeta->endAccess();
          postCrCovAlphaRho->endAccess();
          postCrCovBetaRho->endAccess();
          seed0->endAccess();
          seed1->endAccess();
          seed2->endAccess();

          // time(&timeend);
          // printf("Simulation in FFT domain in %ld seconds \n",timeend-timestart);
          // time(&timestart);

          seed0->setAccessMode(FFTGrid::RANDOMACCESS);
          seed0->invFFTInPlace();

          seed1->setAccessMode(FFTGrid::RANDOMACCESS);
          seed1->invFFTInPlace();


          seed2->setAccessMode(FFTGrid::RANDOMACCESS);
          seed2->invFFTInPlace();

          if(modelAVOdynamic_->getUseLocalNoise()==true)
          {
            float alpha,beta, rho;
            float alphanew, betanew, rhonew;

            for(j=0;j<ny_;j++)
              for(i=0;i<nx_;i++)
                for(k=0;k<nz_;k++)
                {
                  alpha = seed0->getRealValue(i,j,k);
                  beta = seed1->getRealValue(i,j,k);
                  rho = seed2->getRealValue(i,j,k);
                  alphanew = float((*sigmamdnew_)(i,j)[0][0]*alpha+ (*sigmamdnew_)(i,j)[0][1]*beta+(*sigmamdnew_)(i,j)[0][2]*rho);
                  betanew = float((*sigmamdnew_)(i,j)[1][0]*alpha+ (*sigmamdnew_)(i,j)[1][1]*beta+(*sigmamdnew_)(i,j)[1][2]*rho);
                  rhonew = float((*sigmamdnew_)(i,j)[2][0]*alpha+ (*sigmamdnew_)(i,j)[2][1]*beta+(*sigmamdnew_)(i,j)[2][2]*rho);
                  seed0->setRealValue(i,j,k,alphanew);
                  seed1->setRealValue(i,j,k,betanew);
                  seed2->setRealValue(i,j,k,rhonew);
                }
          }

          seed0->add(postAlpha_);
          seed0->endAccess();
          seed1->add(postBeta_);
          seed1->endAccess();
          seed2->add(postRho_);
          seed2->endAccess();

          if(kriging == true) {
            double wall2=0.0, cpu2=0.0;
            TimeKit::getTime(wall2,cpu2);
            doPostKriging(seismicParameters, *seed0, *seed1, *seed2);
            Timings::addToTimeKrigingSim(wall2,cpu2);
          }
          ParameterOutput::writeParameters(simbox_, modelGeneral_, modelSettings_, seed0, seed1, seed2,
                                           outputGridsElastic_, fileGrid_, simNr, kriging);
          // time(&timeend);
          // printf("Back transform and write of simulation in %ld seconds \n",timeend-timestart);
    }

    delete seed0;
    delete seed1;
    delete seed2;

    for(l=0;l<3;l++)
      delete  [] ijkPostCov[l];
    delete [] ijkPostCov;
    delete [] ijkSeed;

  }
  Timings::setTimeSimulation(wall,cpu);
  return(0);
}

void
Crava::doPostKriging(SeismicParametersHolder & seismicParameters,
                     FFTGrid                 & postAlpha,
                     FFTGrid                 & postBeta,
                     FFTGrid                 & postRho)
{

  LogKit::WriteHeader("Kriging to wells");

  CovGridSeparated covGridAlpha      (*seismicParameters.GetCovAlpha()      );
  CovGridSeparated covGridBeta       (*seismicParameters.GetCovBeta()       );
  CovGridSeparated covGridRho        (*seismicParameters.GetCovRho()        );
  CovGridSeparated covGridCrAlphaBeta(*seismicParameters.GetCrCovAlphaBeta());
  CovGridSeparated covGridCrAlphaRho (*seismicParameters.GetCrCovAlphaRho() );
  CovGridSeparated covGridCrBetaRho  (*seismicParameters.GetCrCovBetaRho()  );

  KrigingData3D kd(wells_, nWells_, 1); // 1 = full resolution logs

  std::string baseName = "Raw_" + IO::PrefixKrigingData() + IO::SuffixGeneralData();
  std::string fileName = IO::makeFullFileName(IO::PathToInversionResults(), baseName);
  kd.writeToFile(fileName);

  CKrigingAdmin pKriging(*simbox_,
                         kd.getData(), kd.getNumberOfData(),
                         covGridAlpha, covGridBeta, covGridRho,
                         covGridCrAlphaBeta, covGridCrAlphaRho, covGridCrBetaRho,
                         krigingParameter_);

  pKriging.KrigAll(postAlpha, postBeta, postRho, false, modelSettings_->getDebugFlag(), modelSettings_->getDoSmoothKriging());
}

FFTGrid *
Crava::computeSeismicImpedance(FFTGrid * alpha, FFTGrid * beta, FFTGrid * rho, int angle)
{
  FFTGrid * impedance = createFFTGrid();
  impedance->setType(FFTGrid::DATA);
  impedance->createRealGrid();
  impedance->setAccessMode(FFTGrid::WRITE);

  int rnxp  = alpha->getRNxp();
  alpha->setAccessMode(FFTGrid::READ);
  beta->setAccessMode(FFTGrid::READ);
  rho->setAccessMode(FFTGrid::READ);
  for(int k = 0; k < nzp_; k++) {
    for(int j = 0; j < nyp_; j++)
    {
      for(int i = 0; i < rnxp; i++)
      {
        float imp = 0;
        imp += alpha->getNextReal()*A_[angle][0];
        imp += beta->getNextReal()*A_[angle][1];
        imp += rho->getNextReal()*A_[angle][2];

        impedance->setNextReal(imp);
      }
    }
  }
  impedance->endAccess();
  alpha->endAccess();
  beta->endAccess();
  rho->endAccess();
  return(impedance);
}


void
Crava::computeSyntSeismic(FFTGrid * alpha, FFTGrid * beta, FFTGrid * rho)
{
  LogKit::WriteHeader("Compute Synthetic Seismic and Residuals");

  bool fftDomain = alpha->getIsTransformed();
  if(fftDomain == true) {
    alpha->invFFTInPlace();
    beta->invFFTInPlace();
    rho->invFFTInPlace();
  }

  for(int l=0;l<ntheta_;l++) {
    FFTGrid * imp = computeSeismicImpedance(alpha, beta, rho, l);
    imp->setAccessMode(FFTGrid::RANDOMACCESS);
    for(int i=0;i<nx_; i++) {
      for(int j=0;j<ny_;j++) {
        Wavelet1D impVec(0,nz_, nzp_);
        //impVec.setupAsVector();
        int k;
        for(k=0;k<nz_;k++){
          float value = imp->getRealValue(i, j, k, true);
          impVec.setRAmp(value, k);
        }
        //Tapering:
        float fac = 1.0f/static_cast<float>(nzp_-nz_-1);
        for(;k<nzp_;k++) {
          float value = fac*((k-nz_)*impVec.getRAmp(0)+(nzp_-k-1)*impVec.getRAmp(nz_-1));
          impVec.setRAmp(value, k);
        }
        Wavelet1D resultVec(&impVec, Wavelet::FIRSTORDERFORWARDDIFF);
        resultVec.fft1DInPlace();

        Wavelet1D * localWavelet = seisWavelet_[l]->createLocalWavelet1D(i,j);

        float sf = static_cast<float>(simbox_->getRelThick(i, j))*seisWavelet_[l]->getLocalStretch(i,j);

        for(int k=0;k<(nzp_/2 +1);k++) {
          fftw_complex r = resultVec.getCAmp(k);
          fftw_complex w = localWavelet->getCAmp(k,static_cast<float>(sf));// returns complex conjugate
          fftw_complex s;
          s.re = r.re*w.re+r.im*w.im; //Use complex conjugate of w
          s.im = -r.re*w.im+r.im*w.re;
          resultVec.setCAmp(s,k);
        }
        delete localWavelet;

        resultVec.invFFT1DInPlace();
        for(int k=0;k<nzp_;k++){
          float value = resultVec.getRAmp(k);
          imp->setRealValue(i, j, k, value, true);
        }
      }
    }
    std::string angle     = NRLib::ToString(thetaDeg_[l],1);
    std::string sgriLabel = " Synthetic seismic for incidence angle "+angle;
    std::string fileName  = IO::PrefixSyntheticSeismicData() + angle;
    if(((modelSettings_->getOutputGridsSeismic() & IO::SYNTHETIC_SEISMIC_DATA) > 0) ||
      (modelSettings_->getForwardModeling() == true))
      imp->writeFile(fileName, IO::PathToSeismicData(), simbox_,sgriLabel);
    if((modelSettings_->getOutputGridsSeismic() & IO::SYNTHETIC_RESIDUAL) > 0) {
      FFTGrid seis(nx_, ny_, nz_, nxp_, nyp_, nzp_);

      std::string fileName = IO::makeFullFileName(IO::PathToSeismicData(), IO::FileTemporarySeismic()+NRLib::ToString(l)+IO::SuffixCrava());
      std::string errText;
      seis.readCravaFile(fileName, errText);
      if(errText == "") {
        seis.setAccessMode(FFTGrid::RANDOMACCESS);
        for(int k=0;k<nz_;k++) {
          for(int j=0;j<ny_;j++) {
            for(int i=0;i<nx_;i++) {
              float residual = seis.getRealValue(i, j, k) - imp->getRealValue(i,j,k);
              imp->setRealValue(i, j, k, residual);
            }
          }
        }
        sgriLabel = "Residual computed from synthetic seismic for incidence angle "+angle;
        fileName = IO::PrefixSyntheticResiduals() + angle;
        imp->writeFile(fileName, IO::PathToSeismicData(), simbox_,sgriLabel);
      }
      else {
        errText += "\nFailed to read temporary stored seismic data.\n";
        LogKit::LogMessage(LogKit::Error,errText);
      }
    }
    delete imp;
  }

  if(fftDomain == true) {
    alpha->fftInPlace();
    beta->fftInPlace();
    rho->fftInPlace();
  }
}


float
Crava::computeWDCorrMVar (Wavelet1D* WD ,fftw_real* corrT)
{
  float var = 0.0;
  int i,j,corrInd;

  for(i=0;i<nzp_;i++)
    for(j=0;j<nzp_;j++)
    {
      corrInd = std::max(i-j,j-i);
      var += WD->getRAmp(i)*corrT[corrInd]*WD->getRAmp(j);
    }
    return var;
}


void
Crava::fillkW_flens(int                     k,
                    NRLib::ComplexVector  & kW,
                    Wavelet              ** seisWavelet)
{
  for(int l = 0; l < ntheta_; l++) {
    double kWR =  static_cast<double>( seisWavelet[l]->getCAmp(k).re );
    double kWI = -static_cast<double>( seisWavelet[l]->getCAmp(k).im ); // adjust for complex conjugate in getCAmp(k)
    kW(l) = std::complex<double>(kWR, kWI);
  }
}

void
Crava::fillkWNorm_flens(int                     k,
                        NRLib::ComplexVector  & kWNorm,
                        Wavelet1D            ** wavelet)
{
  for (int l = 0; l < ntheta_; l++)
  {
    double kWNormR =  static_cast<double>(wavelet[l]->getCAmp(k).re/wavelet[l]->getNorm());
    double kWNormI = -static_cast<double>(wavelet[l]->getCAmp(k).im/wavelet[l]->getNorm()); // // adjust for complex conjugate in getCAmp(k)
    kWNorm(l) = std::complex<double>(kWNormR, kWNormI);
  }
}

void
Crava::fillInverseAbskWRobust_flens(int                     k,
                                    NRLib::ComplexVector  & invkW,
                                    Wavelet1D            ** seisWaveletForNorm)
{
  for(int l = 0; l < ntheta_; l++) {
    fftw_complex value       = seisWaveletForNorm[l]->getCAmp(k);
    fftw_complex valueFine   = seisWaveletForNorm[l]->getCAmp(k, 0.999f);// denser sampling of wavelet

    double       modulus     = static_cast<double>(value.re*value.re + value.im*value.im);
    double       modulusFine = static_cast<double>(valueFine.re*valueFine.re + valueFine.im*valueFine.im);
    double       maxMod      = static_cast<double>(std::max(modulus,modulusFine));

    if(maxMod > 0.0)
    {
      invkW(l) = std::complex<double>(static_cast<double>(1.0/sqrt(maxMod)), 0.0);
    }
    else
    {
      invkW(l) = std::complex<double>(seisWaveletForNorm[l]->getNorm()*nzp_*nzp_*100.0f, 0.0);
    }
  }
}



FFTGrid*
Crava::createFFTGrid()
{
  FFTGrid* fftGrid;

  if(fileGrid_)
    fftGrid = new FFTFileGrid(nx_,ny_,nz_,nxp_,nyp_,nzp_);
  else
    fftGrid = new FFTGrid(nx_,ny_,nz_,nxp_,nyp_,nzp_);

  return(fftGrid);
}

FFTGrid*
Crava::copyFFTGrid(FFTGrid * fftGridOld)
{
  FFTGrid* fftGrid;
  if(fileGrid_)
    fftGrid = new FFTFileGrid(reinterpret_cast<FFTFileGrid*>(fftGridOld));
  else
    fftGrid = new FFTGrid(fftGridOld);
  return(fftGrid);
}


FFTFileGrid*
Crava::copyFFTGrid(FFTFileGrid * fftGridOld)
{
  FFTFileGrid* fftGrid;
  fftGrid =  new FFTFileGrid(fftGridOld);
  return(fftGrid);
}

void
Crava::printEnergyToScreen()
{
  int i;
  LogKit::LogFormatted(LogKit::Low,"\n                       ");
  for(i=0;i < ntheta_; i++) LogKit::LogFormatted(LogKit::Low,"  Seismic %4.1f ",thetaDeg_[i]);
  LogKit::LogFormatted(LogKit::Low,"\n----------------------");
  for(i=0;i < ntheta_; i++) LogKit::LogFormatted(LogKit::Low,"---------------");
  LogKit::LogFormatted(LogKit::Low,"\nObserved data variance :");
  for(i=0;i < ntheta_; i++) LogKit::LogFormatted(LogKit::Low,"    %1.3e  ",dataVariance_[i]);
  LogKit::LogFormatted(LogKit::Low,"\nModelled data variance :");
  for(i=0;i < ntheta_; i++) LogKit::LogFormatted(LogKit::Low,"    %1.3e  ",signalVariance_[i]);
  LogKit::LogFormatted(LogKit::Low,"\nError variance         :");
  for(i=0;i < ntheta_; i++) LogKit::LogFormatted(LogKit::Low,"    %1.3e  ",errorVariance_[i]);
  LogKit::LogFormatted(LogKit::Low,"\nWavelet scale          :");
  for(i=0;i < ntheta_; i++) LogKit::LogFormatted(LogKit::Low,"    %2.3e  ",seisWavelet_[i]->getScale());
  LogKit::LogFormatted(LogKit::Low,"\nEmpirical S/N          :");
  for(i=0;i < ntheta_; i++) LogKit::LogFormatted(LogKit::Low,"    %5.2f      ",empSNRatio_[i]);
  LogKit::LogFormatted(LogKit::Low,"\nModelled S/N           :");
  for(i=0;i < ntheta_; i++) LogKit::LogFormatted(LogKit::Low,"    %5.2f      ",theoSNRatio_[i]);
  LogKit::LogFormatted(LogKit::Low,"\n");
}


void
Crava::computeFaciesProb(SpatialWellFilter       * filteredlogs,
                         bool                      useFilter,
                         SeismicParametersHolder & seismicParameters)
{
  ModelSettings * modelSettings = modelSettings_;

  if(modelSettings->getEstimateFaciesProb())
  {
    LogKit::WriteHeader("Facies probability volumes");

    double wall=0.0, cpu=0.0;
    TimeKit::getTime(wall,cpu);

    std::vector<std::string> facies_names = modelGeneral_->getFaciesNames();
    int nfac = static_cast<int>(facies_names.size());
    if(modelGeneral_->getPriorFacies().size() != 0){
      LogKit::LogFormatted(LogKit::Low,"\nPrior facies probabilities:\n");
      LogKit::LogFormatted(LogKit::Low,"\n");
      LogKit::LogFormatted(LogKit::Low,"Facies         Probability\n");
      LogKit::LogFormatted(LogKit::Low,"--------------------------\n");
      const std::vector<float> priorFacies = modelGeneral_->getPriorFacies();
      for(int i=0 ; i<nfac; i++) {
        LogKit::LogFormatted(LogKit::Low,"%-15s %10.4f\n",facies_names[i].c_str(),priorFacies[i]);
      }
    }
    if (simbox_->getdz() > 4.01f) { // Require this density for estimation of facies probabilities
      LogKit::LogFormatted(LogKit::Low,"\nWARNING: The minimum sampling density is lower than 4.0. The FACIES PROBABILITIES\n");
      LogKit::LogFormatted(LogKit::Low,"         generated by CRAVA are not reliable. To get more reliable probabilities    \n");
      LogKit::LogFormatted(LogKit::Low,"         the number of layers must be increased.                                    \n");
      std::string text("");
      text += "Increase the number of layers to improve the quality of the facies probabilities.\n";
      text += "   The minimum sampling density is "+NRLib::ToString(simbox_->getdz())+", and it should be lower than 4.0.\n";
      text += "   To obtain the desired density, the number of layers should be at least "+NRLib::ToString(static_cast<int>(ceil(simbox_->GetLZ()/4.0)))+"\n";
      TaskList::addTask(text);
    }

    if(modelSettings->getFaciesProbFromRockPhysics() == false) {
      LogKit::LogFormatted(LogKit::Low,"\n");
      LogKit::LogFormatted(LogKit::Low,"Well                    Use    SyntheticVs    Deviated\n");
      LogKit::LogFormatted(LogKit::Low,"------------------------------------------------------\n");
      for(int i=0 ; i<nWells_ ; i++) {
        LogKit::LogFormatted(LogKit::Low,"%-23s %3s        %3s          %3s\n",
          wells_[i]->getWellname().c_str(),
          ( wells_[i]->getUseForFaciesProbabilities() ? "yes" : " no" ),
          ( wells_[i]->hasSyntheticVsLog()            ? "yes" : " no" ),
          ( wells_[i]->isDeviated()                   ? "yes" : " no" ));
      }
    }

    std::string baseName = IO::PrefixFaciesProbability();

    FFTGrid * likelihood = NULL;
    if ((modelSettings->getOutputGridsOther() & IO::FACIES_LIKELIHOOD) > 0) {
      int nx = postAlpha_->getNx();
      int ny = postAlpha_->getNy();
      int nz = postAlpha_->getNz();
      if (postAlpha_->isFile())
        likelihood = new FFTFileGrid(nx, ny, nz, nx, ny, nz);
      else
        likelihood = new FFTGrid(nx, ny, nz, nx, ny, nz);
      likelihood->createRealGrid(false);
    }

    if (modelSettings->getFaciesProbRelative())
    {
      meanAlpha_->subtract(postAlpha_);
      meanAlpha_->changeSign();
      meanBeta_->subtract(postBeta_);
      meanBeta_->changeSign();
      meanRho_->subtract(postRho_);
      meanRho_->changeSign();
      if (modelSettings->getFaciesProbFromRockPhysics())
        baseName += "Rock_Physics_";

      std::vector<double> trend_min;
      std::vector<double> trend_max;
      float corrGradI, corrGradJ;
      modelGeneral_->getCorrGradIJ(corrGradI, corrGradJ);
      FindSamplingMinMax(modelGeneral_->getTrendCubes().GetTrendCubeSampling(), trend_min, trend_max);

      fprob_ = new FaciesProb(meanAlpha_,
                              meanBeta_,
                              meanRho_,
                              nfac,
                              modelSettings->getPundef(),
                              modelGeneral_->getPriorFacies(),
                              modelGeneral_->getPriorFaciesCubes(),
                              likelihood,
                              modelGeneral_->getRockDistributionTime0(),
                              modelGeneral_->getFaciesNames(),
                              modelAVOstatic_->getFaciesEstimInterval(),
                              this,
                              seismicParameters,
                              modelAVOdynamic_->getLocalNoiseScales(),
                              modelSettings_,
                              filteredlogs,
                              wells_,
                              modelGeneral_->getTrendCubes(),
                              nWells_,
                              simbox_->getdz(),
                              useFilter,
                              true,
                              trend_min[0],
                              trend_max[0],
                              trend_min[1],
                              trend_max[1]);
    }
    else if(modelSettings->getFaciesProbRelative() == false)
    {
      baseName += "Absolute_";
      if(modelSettings->getFaciesProbFromRockPhysics())
        baseName += "Rock_Physics_";

      std::vector<double> trend_min;
      std::vector<double> trend_max;
      FindSamplingMinMax(modelGeneral_->getTrendCubes().GetTrendCubeSampling(), trend_min, trend_max);

      fprob_ = new FaciesProb(postAlpha_,
                              postBeta_,
                              postRho_,
                              nfac,
                              modelSettings->getPundef(),
                              modelGeneral_->getPriorFacies(),
                              modelGeneral_->getPriorFaciesCubes(),
                              likelihood,
                              modelGeneral_->getRockDistributionTime0(),
                              modelGeneral_->getFaciesNames(),
                              modelAVOstatic_->getFaciesEstimInterval(),
                              this,
                              seismicParameters,
                              modelAVOdynamic_->getLocalNoiseScales(),
                              modelSettings_,
                              filteredlogs,
                              wells_,
                              modelGeneral_->getTrendCubes(),
                              nWells_,
                              simbox_->getdz(),
                              useFilter,
                              false,
                              trend_min[0],
                              trend_max[0],
                              trend_min[1],
                              trend_max[1]);

    }
    if (!modelSettings->getFaciesProbFromRockPhysics()){
      fprob_->calculateConditionalFaciesProb(wells_,
                                             nWells_,
                                             modelAVOstatic_->getFaciesEstimInterval(),
                                             facies_names,
                                             simbox_->getdz());
    }
    LogKit::LogFormatted(LogKit::Low,"\nProbability cubes done\n");


    if (modelSettings->getOutputGridsOther() & IO::FACIESPROB_WITH_UNDEF){
      for (int i=0;i<nfac;i++)
      {
        FFTGrid * grid = fprob_->getFaciesProb(i);
        std::string fileName = baseName +"With_Undef_"+ facies_names[i];
        ParameterOutput::writeToFile(simbox_, modelGeneral_, modelSettings_, grid, fileName,"");
      }
      FFTGrid * grid = fprob_->getFaciesProbUndef();
      std::string fileName = baseName + "Undef";
      ParameterOutput::writeToFile(simbox_, modelGeneral_, modelSettings_, grid, fileName,"");
    }

    fprob_->calculateFaciesProbGeomodel(modelGeneral_->getPriorFacies(),
                                        modelGeneral_->getPriorFaciesCubes());

    if (modelSettings->getOutputGridsOther() & IO::FACIESPROB){
      for (int i=0;i<nfac;i++)
      {
        FFTGrid * grid = fprob_->getFaciesProb(i);
        std::string fileName = baseName + facies_names[i];
        ParameterOutput::writeToFile(simbox_, modelGeneral_, modelSettings_, grid, fileName,"");
      }
    }
    if (modelSettings->getFaciesProbFromRockPhysics() == false) {
      fprob_->writeBWFaciesProb(wells_, nWells_);
      std::vector<double> pValue = fprob_->calculateChiSquareTest(wells_, nWells_, modelAVOstatic_->getFaciesEstimInterval());

      if (modelSettings->getOutputGridsOther() & IO::SEISMIC_QUALITY_GRID)
        QualityGrid qualityGrid(pValue, wells_, simbox_, modelSettings, modelGeneral_);
    }

    if (likelihood != NULL) {
      for(int i=0;i<nfac;i++) {
        FFTGrid * grid = fprob_->createLHCube(likelihood, i,
                                              modelGeneral_->getPriorFacies(), modelGeneral_->getPriorFaciesCubes());
        std::string fileName = IO::PrefixLikelihood() + facies_names[i];
        ParameterOutput::writeToFile(simbox_, modelGeneral_, modelSettings_, grid,fileName,"");
        delete grid;
      }
    }

    Timings::setTimeFaciesProb(wall,cpu);
  }
}

NRLib::Matrix
Crava::getPriorVar0() const
{
  return priorVar0_;
}

NRLib::Matrix
Crava::getPostVar0(void) const
{
  return postVar0_;
}

NRLib::SymmetricMatrix
Crava::getSymmetricPriorVar0() const
{
  NRLib::SymmetricMatrix PriorVar0(3);
  PriorVar0(0,0) = static_cast<double>(priorVar0_(0,0));
  PriorVar0(0,1) = static_cast<double>(priorVar0_(0,1));
  PriorVar0(0,2) = static_cast<double>(priorVar0_(0,2));
  PriorVar0(1,1) = static_cast<double>(priorVar0_(1,1));
  PriorVar0(1,2) = static_cast<double>(priorVar0_(1,2));
  PriorVar0(2,2) = static_cast<double>(priorVar0_(2,2));
  return PriorVar0;
}

NRLib::SymmetricMatrix
Crava::getSymmetricPostVar0(void) const
{
  NRLib::SymmetricMatrix PostVar0(3);
  PostVar0(0,0) = static_cast<double>(postVar0_(0,0));
  PostVar0(0,1) = static_cast<double>(postVar0_(0,1));
  PostVar0(0,2) = static_cast<double>(postVar0_(0,2));
  PostVar0(1,1) = static_cast<double>(postVar0_(1,1));
  PostVar0(1,2) = static_cast<double>(postVar0_(1,2));
  PostVar0(2,2) = static_cast<double>(postVar0_(2,2));
  return PostVar0;
}

//-------------------------------------------
void Crava::computeG(NRLib::Matrix & G) const
//-------------------------------------------
{
  //
  // Class variables in use
  //
  double ** errThetaCov  = errThetaCov_;
  int       n_theta      = ntheta_;

  NRLib::Matrix ErrThetaCov(n_theta, n_theta);
  NRLib::SetMatrixFrom2DArray(ErrThetaCov, errThetaCov);

  NRLib::Matrix Sm     = priorVar0_;
  NRLib::Matrix Smd    = postVar0_;
  NRLib::Matrix Sdelta = Sm - Smd;

  NRLib::Vector Eval(3);
  NRLib::Matrix Evec(3,3);
  NRLib::Matrix EvalMat(3,3);

  NRLib::ComputeEigenVectors(Sdelta, Eval, Evec);

  NRLib::InitializeMatrix(EvalMat, 0.0);
  for (int i=0 ; i < 3 ; i++) {
    if (Eval(i) > 0.0) {
      EvalMat(i,i) = Eval(i);
    }
  }

  NRLib::Matrix EvecT, H;

  EvecT  = NRLib::transpose(Evec);
  H      = Evec * EvalMat;
  Sdelta = H * EvecT;

  NRLib::ComputeEigenVectors(Sm, Eval, Evec);

  NRLib::InitializeMatrix(EvalMat, 0.0);
  for (int i=0 ; i < 3 ; i++) {
    if (Eval(i) > 0.0000001) {
      EvalMat(i, i) = 1.0/sqrt(Eval(i));
    }
  }

  NRLib::Matrix Sinv, A;

  EvecT = flens::transpose(Evec);
  H     = Evec * EvalMat;
  Sinv  = H * EvecT;
  H     = Sinv * Sdelta;
  A     = H * Sinv;

  NRLib::ComputeEigenVectors(A, Eval, Evec);
  NRLib::Sort3x3(Eval, Evec);

  NRLib::Matrix Lg(n_theta, 3);

  NRLib::InitializeMatrix(Lg, 0.0);
  for (int i=0 ; i < std::min(n_theta, 3) ; i++) {
    Lg(i,i) = sqrt(Eval(i)/(1.0 - Eval(i)));
  }

  NRLib::Vector Evale(n_theta);
  NRLib::Matrix Evece(n_theta, n_theta);
  NRLib::Matrix EvalMate(n_theta, n_theta);

  NRLib::ComputeEigenVectors(ErrThetaCov, Evale, Evece);

  NRLib::InitializeMatrix(EvalMate, 0.0);
  for (int i=0 ; i < n_theta ; i++) {
    if(Evale(i) > 0.0) {
      EvalMate(i,i) = sqrt(Evale(i));
    }
  }

  NRLib::Matrix EveceT = flens::transpose(Evece);
  NRLib::Matrix He = Evece * EvalMate;
  Evece = He * EveceT;
  NRLib::Matrix H1 = Evece * Lg;
  EvecT = flens::transpose(Evec);
  NRLib::Matrix H2 = H1 * EvecT;
  G = H2*Sinv;
}

void Crava::newPosteriorCovPointwise(NRLib::Matrix & sigmanew,
                                     NRLib::Matrix & G,
                                     NRLib::Vector & scales,
                                     NRLib::Matrix & sigmamdnew) const
{
  //  this function name is not suited... it returns not what we should think perhaps...
  //  sigmanew=  sqrt( (sigmaM - sigmaM|d_new )^-1 ) * sqrt( (sigmaM -s igmaM|d_old )^-1)
  //  sigmamdnew = Sqrt( Posterior covariance)

  NRLib::Matrix D = NRLib::ZeroMatrix(ntheta_);
  for (int i=0 ; i<ntheta_ ; i++) {
    D(i,i) = sqrt(scales(i));
  }

  NRLib::Matrix ErrThetaCov(ntheta_, ntheta_);
  NRLib::SetMatrixFrom2DArray(ErrThetaCov, errThetaCov_);

  NRLib::Matrix help      = D * ErrThetaCov;
  NRLib::Matrix sigmaenew = help * D;

  NRLib::Matrix GT        = NRLib::transpose(G);
  NRLib::Matrix sigmam    = priorVar0_;
  NRLib::Matrix H1        = G * sigmam;

  help = H1 * GT;
  help = help + sigmaenew;

  NRLib::Vector eigvale(ntheta_);
  NRLib::Matrix eigvece(ntheta_,ntheta_);
  NRLib::ComputeEigenVectors(help, eigvale, eigvece);

  NRLib::Matrix eigvalmate = NRLib::ZeroMatrix(ntheta_);

  for (int i=0 ; i<ntheta_ ; i++) {
    if (eigvale(i) > 0.00000001) {
      eigvalmate(i,i) = 1.0/eigvale(i);
    }
  }

  NRLib::Matrix eigvecetrans = NRLib::transpose(eigvece);
  help    = eigvece * eigvalmate;
  eigvece = help * eigvecetrans;   // eigvece = (G*Sigmam*GT+sigmaE_New)^-1

  NRLib::Matrix H4       = sigmam * GT;
  NRLib::Matrix H2       = H4 * eigvece; // SigmaM*GT*(G*Sigmam*GT+sigmaE_New)^-1
  NRLib::Matrix H3       = H2 * G;       // SigmaM*GT*(G*Sigmam*GT+sigmaE_New)^-1*G
  NRLib::Matrix deltanew = H3 * sigmam;  // SigmaM*GT*(G*Sigmam*GT+sigmaE_New)^-1*GSigmaM

  sigmamdnew = sigmam - deltanew;        // SigmaM - SigmaM*GT*(G*Sigmam*GT+sigmaE_New)^-1*GSigmaM;  is the local posterior covariance.

  NRLib::Vector eigval(3);
  NRLib::Matrix eigvec(3, 3);
  NRLib::ComputeEigenVectors(sigmamdnew, eigval, eigvec);

  NRLib::Matrix eigvalmat = NRLib::ZeroMatrix(3);
  for (int i=0 ; i<3 ; i++) {
    if (eigval(i) > 0.0) {
      eigvalmat(i,i) = std::sqrt(eigval(i));
    }
  }

  NRLib::Matrix eigvectrans = NRLib::transpose(eigvec);

  H3          = eigvec * eigvalmat;
  sigmamdnew = H3 * eigvectrans;      // sigmamdnew = Sqrt( Posterior covariance)

  // -------

  NRLib::ComputeEigenVectors(deltanew, eigval, eigvec);

  NRLib::InitializeMatrix(eigvalmat, 0.0);
  for (int i=0 ; i<3 ; i++) {
    if (eigval(i) > 0.0) {
      eigvalmat(i,i) = std::sqrt(eigval(i));
    }
  }

  H3          = eigvec * eigvalmat;
  eigvectrans = NRLib::transpose(eigvec);
  deltanew    = H3 * eigvectrans;             // sqrt(SigmaM*GT*(G*Sigmam*GT+sigmaE_New)^-1*GSigmaM)

  // -------

  NRLib::Matrix sigmamd    = postVar0_;
  NRLib::Matrix sigmadelta = sigmam - sigmamd;

  NRLib::ComputeEigenVectors(sigmadelta, eigval, eigvec);

  NRLib::InitializeMatrix(eigvalmat, 0.0);
  for (int i=0 ; i<3 ; i++) {
    if (eigval(i) > 0.0000001) {
      eigvalmat(i,i) = 1.0/std::sqrt(eigval(i));
    }
  }

  H3          = eigvec * eigvalmat;
  eigvectrans = NRLib::transpose(eigvec);
  sigmadelta  = H3 * eigvectrans;           // sqrt( sigmaM-sigmaM|d ) (robustified )

  // -------

  sigmanew = deltanew * sigmadelta;         // sqrt( (sigmaM - sigmaM|d_new )^-1 ) * sqrt( (sigmaM -s igmaM|d_old )^-1)
}


NRLib::Matrix
Crava::computeFilter(NRLib::SymmetricMatrix & Sprior,
                     NRLib::SymmetricMatrix & Spost) const
{
  //
  // Filter = I - Sigma_post * inv(Sigma_prior)
  //
  int n = Sprior.dim();

  NRLib::Matrix I = NRLib::IdentityMatrix(n);
  NRLib::Matrix J = I;
  NRLib::CholeskySolve(Sprior, I);
  NRLib::Matrix F = Spost * I;

  F  = F * (-1);
  F += J;

  return F;
}


void Crava::correctAlphaBetaRho(ModelSettings * modelSettings)
{
  int i,j,k;

  NRLib::Matrix G(ntheta_, 3);

  computeG(G);

  NRLib::Matrix sigmanew(3,3);
  NRLib::Matrix sigmamd(3,3);

  double **sigmamdx = new double*[3];
  for(i=0;i<3;i++)
    sigmamdx[i] = new double[3];

  double **sigmamdold = new double*[3];
  for(i=0;i<3;i++)
    sigmamdold[i] = new double[3];

  for(i=0;i<3;i++)
    for(j=0;j<3;j++)
      sigmamdold[i][j] = postVar0_(i,j);

  double  * eigval       = new double[3];
  double ** eigvalmat    = new double*[3];
  double ** eigvec       = new double*[3];
  double ** eigvectrans  = new double*[3];
  int     * error        = new int[1];
  double ** help         = new double*[3];
  for(i=0;i<3;i++)
  {
    eigvec[i]      = new double[3];
    eigvalmat[i]   = new double[3];
    eigvectrans[i] = new double[3];
    help[i]        = new double[3];
  }

  lib_matr_eigen(sigmamdold,3,eigvec,eigval,error);
  for(i=0;i<3;i++)
    for(j=0;j<3;j++)
      if(i==j && eigval[i]>0.0)
        eigvalmat[i][j]=1.0/sqrt(eigval[i]);
      else
        eigvalmat[i][j] = 0.0;
  lib_matr_prod(eigvec,eigvalmat,3,3,3,help);
  lib_matrTranspose(eigvec,3,3,eigvectrans);
  lib_matr_prod(help,eigvectrans,3,3,3,sigmamdold);

  float * alpha     = new float[nz_];
  float * beta      = new float[nz_];
  float * rho       = new float[nz_];
  float * meanalpha = new float[nz_];
  float * meanbeta  = new float[nz_];
  float * meanrho   = new float[nz_];

  if(modelSettings->getNumberOfSimulations()>0)
    sigmamdnew_ = new NRLib::Grid2D<double **>(nx_,ny_,NULL);
  else
    sigmamdnew_ = NULL;

  std::vector<double> minScale(modelAVOdynamic_->getNumberOfAngles());

  for(int angle=0;angle<modelAVOdynamic_->getNumberOfAngles();angle++)
    minScale[angle] = modelAVOdynamic_->getLocalNoiseScale(angle)->FindMin(RMISSING);

  postAlpha_->setAccessMode(FFTGrid::RANDOMACCESS);
  postBeta_->setAccessMode(FFTGrid::RANDOMACCESS);
  postRho_->setAccessMode(FFTGrid::RANDOMACCESS);
  meanAlpha_->setAccessMode(FFTGrid::RANDOMACCESS);
  meanBeta_->setAccessMode(FFTGrid::RANDOMACCESS);
  meanRho_->setAccessMode(FFTGrid::RANDOMACCESS);

  for(i=0;i<nx_;i++)
  {
    for(j=0;j<ny_;j++)
    {
      NRLib::Vector scales(modelAVOdynamic_->getNumberOfAngles());
      for (int angle=0 ; angle<modelAVOdynamic_->getNumberOfAngles() ; angle++)
        scales(angle) = (*(modelAVOdynamic_->getLocalNoiseScale(angle)))(i, j)/minScale[angle];

      newPosteriorCovPointwise(sigmanew,
                               G,
                               scales,
                               sigmamd);

      NRLib::Set2DArrayFromMatrix(sigmamd, sigmamdx);

      lib_matr_prod(sigmamdx,sigmamdold,3,3,3,eigvec); // store product in eigvec

      if(sigmamdnew_!=NULL)
      {
        (*sigmamdnew_)(i,j) = new double*[3];
        for(int ii=0;ii<3;ii++)
        {
          (*sigmamdnew_)(i,j)[ii] = new double[3];
          for(int jj=0;jj<3;jj++)
            (*sigmamdnew_)(i,j)[ii][jj] = eigvec[ii][jj];
        }
      }

      postAlpha_->getRealTrace(alpha, i, j);
      postBeta_->getRealTrace(beta, i, j);
      postRho_->getRealTrace(rho, i, j);
      meanAlpha_->getRealTrace(meanalpha, i, j);
      meanBeta_->getRealTrace(meanbeta, i, j);
      meanRho_->getRealTrace(meanrho, i, j);

      for(k=0;k<nz_;k++)
      {
        float alphadiff = alpha[k] - meanalpha[k];
        float betadiff  = beta[k]  - meanbeta[k];
        float rhodiff   = rho[k]   - meanrho[k];
        alpha[k]  = float(meanalpha[k]+sigmanew(0,0)*alphadiff + sigmanew(0,1)*betadiff + sigmanew(0,2)*rhodiff);
        beta[k]   = float(meanbeta[k] +sigmanew(1,0)*alphadiff + sigmanew(1,1)*betadiff + sigmanew(1,2)*rhodiff);
        rho[k]    = float(meanrho[k]  +sigmanew(2,0)*alphadiff + sigmanew(2,1)*betadiff + sigmanew(2,2)*rhodiff);
      }
      postAlpha_->setRealTrace(i,j, alpha);
      postBeta_->setRealTrace(i,j,beta);
      postRho_->setRealTrace(i,j,rho);
    }
  }

  postAlpha_->endAccess();
  postBeta_->endAccess();
  postRho_->endAccess();
  meanAlpha_->endAccess();
  meanBeta_->endAccess();
  meanRho_->endAccess();

  delete [] alpha;
  delete [] beta;
  delete [] rho;
  delete [] meanalpha;
  delete [] meanbeta;
  delete [] meanrho;

  for(i=0;i<3;i++)
  {
    delete [] eigvectrans[i];
    delete [] eigvalmat[i];
    delete [] eigvec[i];
    delete [] sigmamdold[i];
    delete [] help[i];
  }
  delete [] eigvectrans;
  delete [] eigvalmat;
  delete [] eigval;
  delete [] eigvec;
  delete [] sigmamdold;
  delete [] error;
  delete [] help;

  for(i=0;i<3;i++)
    delete sigmamdx[i];
  delete [] sigmamdx;

}

void Crava::writeBWPredicted(void)
{
  int i;
  for (i=0; i<nWells_; i++)
  {
    BlockedLogs  * bw = wells_[i]->getBlockedLogsOrigThick();

    postAlpha_->setAccessMode(FFTGrid::RANDOMACCESS);
    bw->setLogFromGrid(postAlpha_,0,1,"ALPHA_PREDICTED");
    postAlpha_->endAccess();

    postBeta_->setAccessMode(FFTGrid::RANDOMACCESS);
    bw->setLogFromGrid(postBeta_,0,1,"BETA_PREDICTED");
    postBeta_->endAccess();

    postRho_->setAccessMode(FFTGrid::RANDOMACCESS);
    bw->setLogFromGrid(postRho_,0,1,"RHO_PREDICTED");
    postRho_->endAccess();
   }
}
