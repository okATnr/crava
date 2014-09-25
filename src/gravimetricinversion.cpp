/***************************************************************************
*      Copyright (C) 2008 by Norwegian Computing Center and Statoil        *
***************************************************************************/

//#include "rfftw.h"

#include "src/gravimetricinversion.h"
#include "src/modelgeneral.h"
#include "src/modelgravitystatic.h"
#include "src/modelgravitydynamic.h"
#include "src/fftgrid.h"
#include "src/fftfilegrid.h"
#include "src/simbox.h"
#include "src/definitions.h"
#include "src/io.h"
#include "src/parameteroutput.h"

#include "lib/timekit.hpp"
//#include "lib/lib_matr.h"

#include "nrlib/iotools/logkit.hpp"
#include "nrlib/flens/nrlib_flens.hpp"

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <time.h>
#include <string>
#include <algorithm>

GravimetricInversion::GravimetricInversion(ModelGeneral            *  modelGeneral,
                                           ModelGravityStatic      *  modelGravityStatic,
                                           ModelGravityDynamic     *& modelGravityDynamic,
                                           SeismicParametersHolder &  seismicParameters)
{
  LogKit::WriteHeader("Building Stochastic Model for Gravimetric Inversion");

  double wall=0.0, cpu=0.0;
  TimeKit::getTime(wall,cpu);

  State4D * state4d = modelGeneral->getState4D();

  lag_index_              = modelGravityStatic->GetLagIndex();

  int nxp               = state4d->getMuVpStatic()->getNxp();
  int nyp               = state4d->getMuVpStatic()->getNyp();
  int nzp               = state4d->getMuVpStatic()->getNzp();

  int nx_upscaled       = modelGravityStatic->GetNx_upscaled();
  int ny_upscaled       = modelGravityStatic->GetNy_upscaled();
  int nz_upscaled       = modelGravityStatic->GetNz_upscaled();

  int nxp_upscaled      = modelGravityStatic->GetNxp_upscaled();
  int nyp_upscaled      = modelGravityStatic->GetNyp_upscaled();
  int nzp_upscaled      = modelGravityStatic->GetNzp_upscaled();
  int Np_up             = nxp_upscaled*nyp_upscaled*nzp_upscaled;

  bool   include_level_shift = true;
  double shift_variance    = 0.0; // Initialization, value computed below
  double level_shift        = 0.0;
  int    nObs               = modelGravityDynamic->GetNData();


  FFTGrid * upscaling_kernel_conj = new FFTGrid(modelGravityStatic->GetUpscalingKernel());
  if(upscaling_kernel_conj->getIsTransformed() == false){
    upscaling_kernel_conj->fftInPlace();
  }
  upscaling_kernel_conj->conjugate();  // Conjugate only in FFT domain.

  FFTGrid * upscaling_kernel_abs = new FFTGrid(modelGravityStatic->GetUpscalingKernel());
  upscaling_kernel_abs->abs();

  // Find distribution of rho_current
  FFTGrid * mean_rho_current  = new FFTGrid(seismicParameters.GetMuRho());    // At this stage we are on log scale, \mu_{log rho^c}
  FFTGrid * cov_rho_current   = new FFTGrid(seismicParameters.GetCovRho());   // at this stage: on log scale

  float log_sigma_squared_current = GetSigmaForTransformation(cov_rho_current);
  //float log_mean_current  = GetMeanForTransformation (mean_rho_current);

  //Need to be in real domain from transforming from log domain
  if(mean_rho_current->getIsTransformed())
    mean_rho_current->invFFTInPlace();
  if(cov_rho_current->getIsTransformed())
    cov_rho_current->invFFTInPlace();

  MeanExpTransform(mean_rho_current, log_sigma_squared_current);
  float exp_mean_current =GetMeanForTransformation(mean_rho_current);
  CovExpTransform (cov_rho_current,  exp_mean_current);

  // Compute joint distributions of [mc, ms] = exp([rho_current_log, rho_static_log])
  FFTGrid * mean_rho_static = new FFTGrid(state4d->getMuRhoStatic());
  FFTGrid * cov_rho_static  = new FFTGrid(state4d->getCovRhoRhoStaticStatic());

  float log_sigma_squared_static      = GetSigmaForTransformation(cov_rho_static);
 // float log_mean_static  = GetMeanForTransformation(mean_rho_static);

  //Need to be in real domain from transforming from log domain
  if(mean_rho_static->getIsTransformed())
    mean_rho_static->invFFTInPlace();
  if(cov_rho_static->getIsTransformed())
    cov_rho_static->invFFTInPlace();

  MeanExpTransform(mean_rho_static, log_sigma_squared_static);
  float exp_mean_static =GetMeanForTransformation(mean_rho_static);
  CovExpTransform (cov_rho_static,  exp_mean_static);


  // Cov of exp(rho_static_current_log) cov current = cov static static + cov static dynamic
  FFTGrid * cov_rhorho_static_current = new FFTGrid(state4d->getCovRhoRhoStaticStatic());
  FFTGrid * cov_rhorho_static_dynamic = new FFTGrid(state4d->getCovRhoRhoStaticDynamic());

  if(cov_rhorho_static_current->getIsTransformed())
    cov_rhorho_static_current->invFFTInPlace();

  if(cov_rhorho_static_dynamic->getIsTransformed())
    cov_rhorho_static_dynamic->invFFTInPlace();

  cov_rhorho_static_current->add(cov_rhorho_static_dynamic);
  CrCovExpTransform(cov_rhorho_static_current, exp_mean_static, exp_mean_current);

  // Mean of new parameter
  if(mean_rho_current->getIsTransformed())
    mean_rho_current->invFFTInPlace();




  if(mean_rho_static->getIsTransformed())
    mean_rho_static->invFFTInPlace();

  FFTGrid * mean_rho_change = new FFTGrid(mean_rho_current);
  mean_rho_change->subtract(mean_rho_static);

  ComputeSyntheticGravimetry(mean_rho_change, modelGravityDynamic, 0.0);

  // Covariance of new parameter stepwise computation: Cov_Change = cov_SS + cov_CC- covSC- covCS
  FFTGrid * cov_rho_change = new FFTGrid(cov_rho_static); // Cov_Ch = cov_SS
  cov_rho_change->add(cov_rho_current);// Cov_Ch = cov_SS + cov_CC
  cov_rho_change->subtract(cov_rhorho_static_current);// Cov_Ch = cov_SS + cov_CC- covSC

  cov_rho_change            ->setAccessMode(FFTGrid::READANDWRITE);
  cov_rhorho_static_current->setAccessMode(FFTGrid::READ);

  float value, value2; // Cov_Ch = cov_SS + cov_CC- covSC- covCS
  for(int i = 0; i < nxp; i++){
    for(int j = 0; j < nyp; j++){
      for(int k = 0; k < nzp; k++){
        value  = cov_rho_change->getRealValue(i,j,k,true);
        value2 = cov_rhorho_static_current->getRealValueCyclic(-i, -j, -k);
        cov_rho_change->setRealValue(i, j, k, value-value2,true);
      }
    }
  }
  cov_rho_change           ->endAccess();
  cov_rhorho_static_current->endAccess();
  //cov_rho_change->writeAsciiFile("covRhoChange.dat");

// Now we have a parameter for inversion: meanRhoChange and covRhoChange
  if(mean_rho_change->getIsTransformed() == false)
    mean_rho_change->fftInPlace();

  // Upscale Rho: Convolution in FFT domain and pointwise multiplication
  mean_rho_change->multiply(upscaling_kernel_conj);  // Now is expMeanRhoTotal smoothed

  // Subsample in FFTDomain
  FFTGrid * upscaled_mean_rho_change;
  Subsample(upscaled_mean_rho_change, mean_rho_change,
            nx_upscaled,  ny_upscaled,  nz_upscaled,
            nxp_upscaled, nyp_upscaled, nzp_upscaled);
  int cnx_upscaled=upscaled_mean_rho_change->getCNxp();

  if(upscaled_mean_rho_change->getIsTransformed())
    upscaled_mean_rho_change->invFFTInPlace();

  // Upscale Covariance
  if(cov_rho_change->getIsTransformed() == false)
    cov_rho_change->fftInPlace();

  // Convolution in the FFTdomain;
  cov_rho_change->multiply(upscaling_kernel_abs);  // Now is expCovRhoTotal smoothed
  cov_rho_change->multiply(upscaling_kernel_abs);  // Abs value (or use complex conjugate);

  // Subsample in FFTDomain
  FFTGrid * upscaled_cov_rho_change;
  Subsample(upscaled_cov_rho_change, cov_rho_change,
            nx_upscaled, ny_upscaled, nz_upscaled,
            nxp_upscaled, nyp_upscaled, nzp_upscaled);

  if(upscaled_cov_rho_change->getIsTransformed())
    upscaled_cov_rho_change->invFFTInPlace();
  //upscaled_cov_rho_change->writeAsciiFile("upscaled_cov_rho_change.txt");

  LogKit::WriteHeader("Performing Gravimetric Inversion");

  NRLib::Matrix G = modelGravityDynamic->GetGMatrix();
  ExpandMatrixWithZeros(G, Np_up, include_level_shift);

  NRLib::Vector    Rho(Np_up);
  VectorizeFFTGrid(Rho, upscaled_mean_rho_change);

  NRLib::Matrix            Sigma(Np_up, Np_up);
  NRLib::InitializeMatrix (Sigma, 0.0);
  ReshapeCovAccordingToLag(Sigma, upscaled_cov_rho_change);

  NRLib::Vector      gravity_data(nObs);
  std::vector<float> d  = modelGravityDynamic->GetGravityResponse();
  std::vector<float> d0 = modelGravityStatic->GetGravityResponse();

  NRLib::Matrix           Sigma_error(nObs, nObs);
  NRLib::InitializeMatrix(Sigma_error, 0.0);
  std::vector<float> std_dev = modelGravityDynamic->GetGravityStdDev();

  //spool over data from std vec to NRLib Vector and summing the squares of the data values for use in level shift
    for(int i = 0; i<nObs; i++){
      gravity_data(i)  = d[i]-d0[i];
      shift_variance += std_dev[i]*std_dev[i];
      Sigma_error(i,i) = std_dev[i];
    }

  shift_variance /= nObs;

  if(include_level_shift){
    // Expand prior mean with one element equal to 0
    int l = Rho.length();
    NRLib::Vector RhoNew(l+1);
    for(int i = 0; i<Rho.length(); i++){
      RhoNew(i) = Rho(i);
    }
    RhoNew(l) = 0;   // set last value
    Rho = RhoNew;

    // Expand prior covariance matrix
    ExpandCovMatrixWithLevelShift(Sigma, shift_variance);
  }

  NRLib::WriteMatrixToFile("Sigma_m.txt", Sigma);
  NRLib::WriteMatrixToFile("Sigma_error.txt", Sigma_error);
  NRLib::WriteVectorToFile("Mu.txt", Rho);
  NRLib::WriteVectorToFile("Gravity_data.txt",gravity_data);
  NRLib::WriteVectorToFile("PriorResponce.txt",modelGravityDynamic->GetSyntheticData());

  NRLib::Vector Rho_posterior  (Np_up);
  NRLib::Matrix Sigma_posterior(Np_up, Np_up);

  NRLib::Matrix GT         = NRLib::transpose(G);
  NRLib::Matrix G_Sigma    = G * Sigma;
  NRLib::Matrix G_Sigma_GT = G_Sigma * GT;
  NRLib::Matrix Sigma_GT   = Sigma * GT;

  NRLib::Matrix inv_G_Sigma_GT_plus_Sigma_error = G_Sigma_GT + Sigma_error;
  NRLib::Invert(inv_G_Sigma_GT_plus_Sigma_error);

  NRLib::Vector temp_1 = gravity_data - modelGravityDynamic->GetSyntheticData(); // NBNB bias correction do not use G*Rho but GetSyntheticData()
  NRLib::Vector temp_2 = inv_G_Sigma_GT_plus_Sigma_error * temp_1;
  temp_1               = Sigma_GT*temp_2;
  Rho_posterior        = Rho + temp_1;


  NRLib::Matrix temp_3 = inv_G_Sigma_GT_plus_Sigma_error * G_Sigma;
  NRLib::Matrix temp_4 = Sigma_GT*temp_3;
  Sigma_posterior      = Sigma - temp_4;

  // Remove shift parameter
  if(include_level_shift){
    RemoveLevelShiftFromVector(Rho_posterior, level_shift);
    RemoveLevelShiftFromCovMatrix(Sigma_posterior);
  }
  NRLib::WriteVectorToFile("Rho_posterior.txt", Rho_posterior); NRLib::WriteMatrixToFile("Sigma_posterior.txt", Sigma_posterior);

  // Reshape back to FFTGrid
  ReshapeVectorToFFTGrid(upscaled_mean_rho_change, Rho_posterior);

  // For backsampling and deconvolution, need to be in FFT-domain
  if(mean_rho_change->getIsTransformed()==false)
    mean_rho_change->fftInPlace();

  if(upscaled_mean_rho_change->getIsTransformed()==false)
    upscaled_mean_rho_change->fftInPlace();

  //Backsample and deconvolve in Fourier domain
  Backsample(upscaled_mean_rho_change, mean_rho_change); //Now meanRhoTotal is posterior!
  Divide(mean_rho_change, upscaling_kernel_conj);

  if(mean_rho_change->getIsTransformed())
    mean_rho_change->invFFTInPlace();

  ComputeSyntheticGravimetry(mean_rho_change, modelGravityDynamic, level_shift);


  /// Posterior upscaled covariance
  FFTGrid * posterior_upscaled_cov_rho_change = new FFTGrid(nx_upscaled, ny_upscaled, nz_upscaled,
                                                           nxp_upscaled, nyp_upscaled, nzp_upscaled);
  posterior_upscaled_cov_rho_change->createRealGrid();
  posterior_upscaled_cov_rho_change->setType(FFTGrid::COVARIANCE);

  ReshapeCovMatrixToFFTGrid(posterior_upscaled_cov_rho_change, Sigma_posterior);


  // Odds algorithm
  // Pick elements that corresponds to effect of inversion, not from adjusting to pos def matrix, in FFT domain
  FFTGrid * fft_factor = new FFTGrid(nx_upscaled,  ny_upscaled,  nz_upscaled,
                                     nxp_upscaled, nyp_upscaled, nzp_upscaled);
  fft_factor->createRealGrid();
  fft_factor->setType(FFTGrid::OPERATOR);
  fft_factor->fillInConstant(0.0f);
  fft_factor->fftInPlace();


  // Need to be in FFT-domain
  if(upscaled_cov_rho_change->getIsTransformed() == false)
    upscaled_cov_rho_change->fftInPlace();
  if(posterior_upscaled_cov_rho_change->getIsTransformed() == false)
    posterior_upscaled_cov_rho_change->fftInPlace();

  float reference;
  double nu = 0.05;
  fftw_complex prior = upscaled_cov_rho_change          ->getFirstComplexValue();
  fftw_complex post  = posterior_upscaled_cov_rho_change->getFirstComplexValue();
  reference =  post.re/prior.re*nu;

  fft_factor                      ->setAccessMode(FFTGrid::WRITE);
  upscaled_cov_rho_change          ->setAccessMode(FFTGrid::READ);
  posterior_upscaled_cov_rho_change->setAccessMode(FFTGrid::READ);
  fftw_complex one;
  one.re=1.0f;
  one.im=0.0f;
  for(int k = 0; k < nz_upscaled; k++){
    for(int j = 0; j < ny_upscaled; j++){
      for(int i = 0; i < cnx_upscaled; i++){
        prior = upscaled_cov_rho_change          ->getComplexValue(i,j,k,true);
        post  = posterior_upscaled_cov_rho_change->getComplexValue(i,j,k,true);

        float m;
        m = post.re/ prior.re;

        if(m > reference){
          fftw_complex ratio;
          ratio.re = posterior_upscaled_cov_rho_change->getComplexValue(i,j,k).re/upscaled_cov_rho_change->getComplexValue(i,j,k,true).re;
          ratio.im=0.0;

          if(ratio.re < 1)
            fft_factor->setComplexValue(i, j, k, ratio,true);
          else
            fft_factor->setComplexValue(i, j, k, one,true);
        }
        else{
          fft_factor->setComplexValue(i, j, k, one,true);
        }
      }
    }
  }
  fft_factor                      ->endAccess();
  upscaled_cov_rho_change          ->endAccess();
  posterior_upscaled_cov_rho_change->endAccess();

  posterior_upscaled_cov_rho_change = upscaled_cov_rho_change; // Set posterior equal to prior and introduce effects of inversion through factors in fft_factor-grid

  posterior_upscaled_cov_rho_change->multiply(fft_factor);  // pointwise multiplication in complex domain

  // For backsampling and deconvolution need to be in Fourier domain
   if(cov_rho_change->getIsTransformed() == false)
    cov_rho_change->fftInPlace();

   if(posterior_upscaled_cov_rho_change->getIsTransformed() == false)
     posterior_upscaled_cov_rho_change->fftInPlace();

  //Backsample in Fourier domain
  Backsample(posterior_upscaled_cov_rho_change, cov_rho_change);  // covRhoTotal is now posterior!

  // Deconvolution: Do pointwise division in FFTdomain - twice
  Divide(cov_rho_change, upscaling_kernel_abs);
  Divide(cov_rho_change, upscaling_kernel_abs);


   // Only for debugging purposes
  if(posterior_upscaled_cov_rho_change->getIsTransformed() == true)
    posterior_upscaled_cov_rho_change->invFFTInPlace();
  NRLib::Matrix Post_sigma_temp(Np_up, Np_up);
  NRLib::InitializeMatrix (Post_sigma_temp, 0.0);
  ReshapeCovAccordingToLag(Post_sigma_temp, posterior_upscaled_cov_rho_change);
  NRLib::WriteMatrixToFile("Sigma_posterior2.txt", Post_sigma_temp);

  // For transforming back to log-domain, need to be in real domain
   if(cov_rho_change->getIsTransformed() == true)
    cov_rho_change->invFFTInPlace();

  // Comput joint distribution of current and static  in exp domain
  mean_rho_change->add(mean_rho_static); // is now  current in exp domain

  // inverting the relation C_dd = C_cc -C_cs-C_sc + C_ss
  // C_cc  = C_dd - C_ss + C_cs + C_sc
  cov_rho_change->subtract(cov_rho_static); // - C_ss
  cov_rho_change->add( cov_rhorho_static_current); // + C_sc
  cov_rho_change            ->setAccessMode(FFTGrid::READANDWRITE);
  cov_rhorho_static_current->setAccessMode(FFTGrid::READ);

//  float value, value2; //  + C_cs
  for(int i = 0; i < nxp; i++){
    for(int j = 0; j < nyp; j++){
      for(int k = 0; k < nzp; k++){
        value  = cov_rho_change->getRealValue(i,j,k,true);
        value2 = cov_rhorho_static_current->getRealValueCyclic(-i, -j, -k);
        cov_rho_change->setRealValue(i, j, k, value+value2,true);
      }
    }
  }
  cov_rho_change           ->endAccess();
  cov_rhorho_static_current->endAccess();
  //cov_rho_change->writeAsciiFile("covRhoChange.dat");

  // Computing properties for "Log World"

  // float exp_mean_change = GetMeanForTransformation(mean_rho_change);
  // Using exp_mean_current from start to match transform of prior
  CovLogTransform(cov_rho_change, exp_mean_current);
  CovLogTransform (cov_rho_static,  exp_mean_static);
  CrCovLogTransform(cov_rhorho_static_current, exp_mean_static, exp_mean_current);


  float log_sigma_squared_change = GetSigmaForTransformation(cov_rho_change);
  MeanLogTransform(mean_rho_change, log_sigma_squared_change);
  MeanLogTransform(mean_rho_static, log_sigma_squared_static);


  // computing dynamic part in  log world
  mean_rho_change->subtract(mean_rho_static);// is now  dynamic change in log  domain

  //  d=c-s
  // C_dd  = C_cc -C_cs-C_sc+C_ss
  cov_rho_change->add(cov_rho_static); // +C_ss
  cov_rho_change->subtract(cov_rhorho_static_current); // -C_sc

  cov_rho_change            ->setAccessMode(FFTGrid::READANDWRITE);// -C_cs anc below
  cov_rhorho_static_current->setAccessMode(FFTGrid::READ);

 // Cov_Ch = cov_SS + cov_CC- covSC- covCS
  for(int i = 0; i < nxp; i++){
    for(int j = 0; j < nyp; j++){
      for(int k = 0; k < nzp; k++){
        value  = cov_rho_change->getRealValue(i,j,k,true);
        value2 = cov_rhorho_static_current->getRealValueCyclic(-i, -j, -k);
        cov_rho_change->setRealValue(i, j, k, value-value2,true);
      }
    }
  }
  cov_rho_change           ->endAccess();
  cov_rhorho_static_current->endAccess();

  // Ready to put back into seismicparametersholder and state4d
  mean_rho_change->fftInPlace();
  cov_rho_change->fftInPlace();
  state4d->updateWithSingleParameter(mean_rho_change,cov_rho_change ,5);

  // Delete all FFTGrids create with "new"
  delete upscaling_kernel_conj;
  delete upscaling_kernel_abs;
  delete mean_rho_current;
  delete cov_rho_current;
  delete mean_rho_static;
  delete cov_rho_static;
  delete cov_rhorho_static_current;
  delete cov_rhorho_static_dynamic;
  delete mean_rho_change;
  delete cov_rho_change;
  delete posterior_upscaled_cov_rho_change;

  delete fft_factor;
}

GravimetricInversion::~GravimetricInversion()
{
}

float
  GravimetricInversion::GetSigmaForTransformation(FFTGrid * sigma)
{
  float sigma_squared = 0;
  if(sigma->getIsTransformed() == false){

    sigma_squared = sigma->getFirstRealValue();   // \sigma^2
  }
  else{
  // Loop though grid in complex domain.
    sigma_squared = 0;

    double realSum=0.0;
    fftw_complex sum;
    sum.re = 0.0;
    sum.im = 0.0;
    int f = 1;

    sigma->setAccessMode(FFTGrid::READ);

    for(int k=0; k<sigma->getNzp(); k++){
      for(int j=0; j<sigma->getNyp(); j++){
        for(int i=0; i<sigma->getCNxp(); i++){
          f=2;
          if(i == 0)                   // the first does not have a complex conjugate
            f = 1;
          if(sigma->getNxp() % 2 == 0){ // if there is an even number the last does not have a complex conjugate
            if(i==sigma->getCNxp()-1)
              f=1;
          }

          fftw_complex value = sigma->getNextComplex();
          sum.re += f*value.re;
          sum.im += f*value.im;   // Blir ikke null, fordi summerer ikke konjugerte par i praksis
          realSum +=f*value.re;
        }
      }
    }
    sigma->endAccess();

    // Due to summing complex conjugate numbers, then imaginary part is zero
    double N = static_cast<double>(sigma->getNxp()*sigma->getNyp()*sigma->getNzp());
    sigma_squared = static_cast<float>(realSum/N);
  }
  return(sigma_squared);
}

float
  GravimetricInversion::GetMeanForTransformation(FFTGrid * grid)
{
  float mean = 0;

  if(grid->getIsTransformed() == true){
    int nxp = grid->getNxp();
    int nyp = grid->getNyp();
    int nzp = grid->getNzp();

    fftw_complex mean_tmp = grid->getFirstComplexValue();          // Standard way of getting mean value of a FFTGrid (including padded region)
    mean                  = mean_tmp.re/pow(static_cast<float>(nxp*nyp*nzp), 0.5f);   // Note the scaling
  }
  else{
    // Loop through grid in real domain
    grid->setAccessMode(FFTGrid::READ);
    double sum = 0.0;
    double partSum;
    for(int k=0;k<grid->getNzp();k++) {
      for(int j=0;j<grid->getNyp();j++) {
        partSum=0.0;
        for(int i=0;i<grid->getNxp();i++) {
          partSum += grid->getNextReal();
        }
        for(int i = 0; i<grid->getRNxp()-grid->getNxp(); i++){
          grid->getNextReal();
        }
        sum += partSum;
      }
    }
    grid->endAccess();

    double N = static_cast<double>(grid->getNxp()*grid->getNyp()*grid->getNzp());
    mean = static_cast<float>(sum/N);

  }

  return(mean);
}

void
  GravimetricInversion::MeanExpTransform(FFTGrid * log_mean, float sigma_squared)
{
  assert(log_mean->getIsTransformed() == false);

  log_mean->addScalar(0.5f*sigma_squared);  // \mu_{log rho^c} + 0.5*\sigma^2
  log_mean->expTransf();  //exp{\mu_{log rho^c} + 0.5*\sigma^2}. Finished transformation.
}

void
  GravimetricInversion::CovExpTransform(FFTGrid  * log_cov, float mean)
{ // Note  mean is mean of exp variable
  //  log_cov is for the log variable
  assert(log_cov->getIsTransformed() == false);

  log_cov->expTransf();
  log_cov->addScalar(-1);
  log_cov->multiplyByScalar(mean*mean);
}

void
  GravimetricInversion::CrCovExpTransform(FFTGrid * log_cov,  float mean_a, float mean_b)
{
  assert(log_cov->getIsTransformed() == false);

  log_cov->expTransf();
  log_cov->addScalar(-1);
  log_cov->multiplyByScalar(mean_a*mean_b);
}

void
  GravimetricInversion::MeanLogTransform(FFTGrid * mean,    float sigma_squared)
{
  assert(mean->getIsTransformed() == false);

  mean->logTransf();
  mean->addScalar(-0.5f*sigma_squared);
}

void
  GravimetricInversion::CovLogTransform(FFTGrid  * cov,     float mean)
{
  assert(cov->getIsTransformed() == false);

  cov->multiplyByScalar(1.0f/(mean*mean));
  cov->addScalar(1);
  cov->logTransf();
}

void
  GravimetricInversion::CrCovLogTransform(FFTGrid * cov,  float mean_a, float mean_b)
{
  assert(cov->getIsTransformed() == false);

  cov->multiplyByScalar(1.0f/(mean_a*mean_b));
  cov->addScalar(1);
  cov->logTransf();


}



void
  GravimetricInversion::Subsample(FFTGrid *& upscaled_grid, FFTGrid * original_grid, int nx_up, int ny_up, int nz_up, int nxp_up, int nyp_up, int nzp_up)
{
  assert(original_grid->getIsTransformed());

  upscaled_grid = new FFTGrid(nx_up, ny_up, nz_up, nxp_up, nyp_up, nzp_up);
  upscaled_grid->createComplexGrid();
  int cnxp_up= upscaled_grid->getCNxp();
  upscaled_grid->setType(original_grid->getType());

  upscaled_grid->setAccessMode(FFTGrid::WRITE);
  original_grid->setAccessMode(FFTGrid::READ);

  int nyp_up_half = static_cast<int>(ceil(static_cast<double>(nyp_up)/2));
  int nzp_up_half = static_cast<int>(ceil(static_cast<double>(nzp_up)/2));
  int nxp = original_grid->getNxp();
  int nyp = original_grid->getNyp();
  int nzp = original_grid->getNzp();

  // Set up index-vectorer
  std::vector<int> y_indices(nyp_up);
  std::vector<int> z_indices(nzp_up);


  for(int i = 0; i<nyp_up; i++){
    if(i<nyp_up_half)
      y_indices[i] = i;
    else
      y_indices[i] = nyp - (nyp_up - i);
  }
  for(int i = 0; i<nzp_up; i++){
    if(i<nzp_up_half)
      z_indices[i] = i;
    else
      z_indices[i] = nzp - (nzp_up - i);
  }

  // Subsample the grid in FFT domain - use also padded region.
  for(int i = 0; i < cnxp_up; i++){
    for(int j = 0; j < nyp_up; j++){
      for(int k = 0; k < nzp_up; k++){
        upscaled_grid->setComplexValue(i, j, k, original_grid->getComplexValue(i, y_indices[j], z_indices[k], true),true);
      }
    }
  }
  upscaled_grid->endAccess();
  original_grid->endAccess();
  if(original_grid->getType()== FFTGrid::PARAMETER)
    upscaled_grid->multiplyByScalar(static_cast<float>(sqrt( static_cast<double>(nxp_up*nyp_up*nzp_up)/static_cast<double>(nxp*nyp*nzp))));
  if(original_grid->getType()== FFTGrid::COVARIANCE)
    upscaled_grid->multiplyByScalar(static_cast<float>(static_cast<double>(nxp_up*nyp_up*nzp_up)/static_cast<double>(nxp*nyp*nzp)));
}

void
  GravimetricInversion::Backsample(FFTGrid * upscaled_grid, FFTGrid * new_full_grid)
{
  assert(upscaled_grid->getIsTransformed());
  assert(new_full_grid->getIsTransformed());

  new_full_grid->setAccessMode(FFTGrid::WRITE);
  upscaled_grid->setAccessMode(FFTGrid::READ);

  int cnxp_up = upscaled_grid->getCNxp();
  int nxp_up = upscaled_grid->getNxp();
  int nyp_up = upscaled_grid->getNyp();
  int nzp_up = upscaled_grid->getNzp();

  int nxp  = new_full_grid->getNxp();
  int nyp  = new_full_grid->getNyp();
  int nzp  = new_full_grid->getNzp();

   if(new_full_grid->getType()== FFTGrid::PARAMETER)
    upscaled_grid->multiplyByScalar(static_cast<float>(sqrt( static_cast<double>(nxp*nyp*nzp)/static_cast<double>(nxp_up*nyp_up*nzp_up))));
  if(new_full_grid->getType()== FFTGrid::COVARIANCE)
    upscaled_grid->multiplyByScalar(static_cast<float>(static_cast<double>(nxp*nyp*nzp)/static_cast<double>(nxp_up*nyp_up*nzp_up)));


  int nyp_up_half = static_cast<int>(ceil(static_cast<double>(nyp_up)/2));
  int nzp_up_half = static_cast<int>(ceil(static_cast<double>(nzp_up)/2));

  // Set up index-vectorer

  std::vector<int> y_indices(nyp_up);
  std::vector<int> z_indices(nzp_up);


  for(int i = 0; i<nyp_up; i++){
    if(i<nyp_up_half)
      y_indices[i] = i;
    else
      y_indices[i] = nyp - (nyp_up - i);
  }
  for(int i = 0; i<nzp_up; i++){
    if(i<nzp_up_half)
      z_indices[i] = i;
    else
      z_indices[i] = nzp - (nzp_up - i);
  }
  // Subsample the grid in FFT domain - use also padded region
  for(int i = 0; i < cnxp_up; i++){
    for(int j = 0; j < nyp_up; j++){
      for(int k = 0; k < nzp_up; k++){
        new_full_grid->setComplexValue(i, y_indices[j], z_indices[k], upscaled_grid->getComplexValue(i,j,k, true));
      }
    }
  }

  new_full_grid->endAccess();
  upscaled_grid->endAccess();
  }

void
  GravimetricInversion::VectorizeFFTGrid(NRLib::Vector &vec, FFTGrid * grid, bool withPadding)
{
  assert(grid->getIsTransformed() == false);

  int nx, ny, nz;
  if(withPadding){
  nx = grid->getNxp();
  ny = grid->getNyp();
  nz = grid->getNzp();
  }
  else{
  nx = grid->getNx();
  ny = grid->getNy();
  nz = grid->getNz();
  }

  grid->setAccessMode(FFTGrid::RANDOMACCESS);

  int I = 0;
  for(int i = 0; i < nx; i++){
    for(int j = 0; j < ny; j++){
      for(int k = 0; k < nz; k++){
        I =  i + j*nx + k*nx*ny;
        vec(I) = grid->getRealValue(i,j,k,true);
      }
    }
  }
  grid->endAccess();
}

void
  GravimetricInversion::ReshapeVectorToFFTGrid(FFTGrid * grid, NRLib::Vector vec)
{
  assert(grid->getIsTransformed() == false);

  grid->setAccessMode(FFTGrid::RANDOMACCESS);

  int nxp = grid->getNxp();
  int nyp = grid->getNyp();
  int nzp = grid->getNzp();
  int I = 0;
  for(int i = 0; i < nxp; i++){
    for(int j = 0; j < nyp; j++){
      for(int k = 0; k < nzp; k++){
        I =  i + j*nxp + k*nxp*nyp;
        grid->setRealValue(i,j,k,static_cast<float>(vec(I)),true);
      }
    }
  }
  grid->endAccess();
}

void
  GravimetricInversion::ReshapeCovAccordingToLag(NRLib::Matrix &CovMatrix, FFTGrid * covGrid)
{
  // Reshape according to lag
  assert(covGrid->getIsTransformed() == false);

  int nxp = covGrid->getNxp();
  int nyp = covGrid->getNyp();
  int nzp = covGrid->getNzp();

  covGrid->setAccessMode(FFTGrid::READ);
  int I;
  int J;
  for(int k1 = 1; k1 <= nzp; k1++){
    for(int j1 = 1; j1 <= nyp; j1++){
      for(int i1 = 1; i1 <= nxp; i1++){
        I =  i1 + (j1-1)*nxp + (k1-1)*nxp*nyp;

        for(int k2 = 1; k2 <= nzp; k2++){
          for(int j2 = 1; j2 <= nyp; j2++){
            for(int i2 = 1; i2 <= nxp; i2++){
              J = i2 + (j2-1)*nxp + (k2-1)*nxp*nyp;

              if(lag_index_[I-1][J-1][0]==-1 && lag_index_[I-1][J-1][1]==-1 && lag_index_[I-1][J-1][2]==-1)
                CovMatrix(I-1,J-1) = 0.0;
              else{
                CovMatrix(I-1,J-1) = covGrid->getRealValue(lag_index_[I-1][J-1][0], lag_index_[I-1][J-1][1], lag_index_[I-1][J-1][2], true);
              }
            }
          }
        }

      }
    }
  }
  covGrid->endAccess();
}

void
  GravimetricInversion::ReshapeCovMatrixToFFTGrid(FFTGrid * cov_grid, NRLib::Matrix cov_matrix)
{
  // Reshape back from covariance matrix to 3D cube

  assert(cov_grid->getIsTransformed() == false);

  int nx = cov_grid->getNx();
  int ny = cov_grid->getNy();
  int nz = cov_grid->getNz();

  int nxp = cov_grid->getNxp();
  int nyp = cov_grid->getNyp();
  int nzp = cov_grid->getNzp();

  // initilize arrays
  std::vector<std::vector<std::vector<float> > > sum;
  std::vector<std::vector<std::vector<int> > >   counter;
  counter.resize(nxp);
  sum    .resize(nxp);
  for (int i = 0; i < nxp; ++i) {
    counter[i].resize(nyp);
    sum[i]    .resize(nyp);
    for (int j = 0; j < nyp; ++j){
      counter[i][j].resize(nzp);
      sum[i][j]    .resize(nzp);
    }
  }

  cov_grid->setAccessMode(FFTGrid::WRITE);
  // Intended: One set of for loops over padded region as well, the other set of for loops over nx, ny, nz
  int I, J;
  int i, j, k;
  for(int k1 = 1; k1 <= nzp; k1++){
    for(int j1 = 1; j1 <= nyp; j1++){
      for(int i1 = 1; i1 <= nxp; i1++){
        I =  i1 + (j1-1)*nxp + (k1-1)*nxp*nyp;
        for(int k2 = 1; k2 <= nz; k2++){
          for(int j2 = 1; j2 <= ny; j2++){
            for(int i2 = 1; i2 <= nx; i2++){
              J = i2 + (j2-1)*nx + (k2-1)*nx*ny;
              if(lag_index_[I-1][J-1][0] >= 0 && lag_index_[I-1][J-1][1] >= 0 && lag_index_[I-1][J-1][2] >= 0){
                i = lag_index_[I-1][J-1][0];
                j = lag_index_[I-1][J-1][1];
                k = lag_index_[I-1][J-1][2];
                sum[i][j][k]    += static_cast<float>(cov_matrix(I-1,J-1));
                counter[i][j][k]++;
              }
            }
          }
        }
      }
    }
  }

  for(int k1 = 0; k1 < nzp; k1++){
    for(int j1 = 0; j1 < nyp; j1++){
      for(int i1 = 0; i1 < nxp; i1++){
        if(counter[i1][j1][k1]>0){
          float value = sum[i1][j1][k1]/counter[i1][j1][k1];
          cov_grid->setRealValue(i1, j1, k1, value,true);
        }
      }
    }
  }
  cov_grid->endAccess();
}

void
  GravimetricInversion::ExpandMatrixWithZeros(NRLib::Matrix &G, int Np, bool include_level_shift)
{
  //Expanding matrix to include padded region

  int r = G.rows().length();
  NRLib::Matrix G_star(r, Np);

  if(include_level_shift)
    G_star.resize(r, Np+1);

  // Initilize to zero
  NRLib::InitializeMatrix(G_star, 0.0);

  // Copy all elements
  for(int i = 0; i<G.rows().length(); i++)
    for(int j = 0; j<G.cols().length(); j++)
    {
      G_star(i,j) = G(i,j);
    }

    if(include_level_shift){
      // Last column with ones
      for(int i = 0; i<G.rows().length(); i++)
        G_star(i,Np) = 1;
    }

    G = G_star;
}

void
  GravimetricInversion::ExpandCovMatrixWithLevelShift(NRLib::Matrix &Sigma, double shift_variance)
{
    int r = Sigma.rows().length();

    NRLib::Matrix Sigma_star(r+1, r+1);
    NRLib::InitializeMatrix(Sigma_star, 0.0);

    // Copy all values except last row and last column - they are left to be zero.
    for(int i = 0; i<r; i++)
      for(int j = 0; j<r; j++){
        Sigma_star(i,j) = Sigma(i,j);
    }
    // Set last element
    Sigma_star(r,r) = shift_variance;

    Sigma = Sigma_star;
}

void
  GravimetricInversion::RemoveLevelShiftFromVector(NRLib::Vector &rho, double &level_shift)
{
    int r       = rho.length();

    // Level shift is found in the last element of the vector
    level_shift = rho(r-1);

    // Copy all elements except last element
    NRLib::Vector rho_new(r-1);
    for(int i = 0; i<r-1; i++){
      rho_new(i) = rho(i);
    }
    rho = rho_new;
}

void
  GravimetricInversion::RemoveLevelShiftFromCovMatrix(NRLib::Matrix &Sigma)
{
  int r = Sigma.rows().length();

  // Copy all elements except last row and last column
  NRLib::Matrix new_Sigma(r-1, r-1);
  for(int i = 0; i<r-1; i++)
    for(int j = 0; j<r-1; j++){
      new_Sigma(i,j) = Sigma(i,j);
    }

  Sigma = new_Sigma;
}



void
GravimetricInversion::Divide(FFTGrid *& fftGrid_numerator, FFTGrid * fftGrid_denominator)
{
  assert(fftGrid_numerator->getNxp() == fftGrid_denominator->getNxp());

  if(fftGrid_numerator->getIsTransformed()==true && fftGrid_denominator->getIsTransformed()==true)
  {
    // Division of complex numbers:
    // \frac{a + bi}{c + di} =
    // \frac{(a + bi)(c - di)}{(c + di)(c - di)} =  // <- Multiply with conjugate denominator
    // \frac{(ac + bd)}{(c^2 + d^2)} + \frac{(bc - ad)}{(c^2 + d^2)}i

    for(int i=0;i<fftGrid_numerator->getcsize();i++)
    {
      fftw_complex numerator   = fftGrid_numerator->getNextComplex();
      fftw_complex denominator = fftGrid_denominator->getNextComplex();;

      fftw_complex tmp;
      tmp.re = numerator.re*denominator.re + numerator.im*denominator.im;
      tmp.im = numerator.im*denominator.re - numerator.re*denominator.im;

      float denominator_tmp = denominator.re*denominator.re + denominator.im*denominator.im;

      if(denominator_tmp != 0 ){
        fftw_complex answer;
        answer.re = tmp.re / denominator_tmp;
        answer.im = tmp.im / denominator_tmp;
        fftGrid_numerator->setNextComplex(answer);
      }
      else{
        // Do nothing when denominator is zero
        fftGrid_numerator->setNextComplex(numerator);
      }
    }
  }

   if(fftGrid_numerator->getIsTransformed()==false && fftGrid_denominator->getIsTransformed()==false)
   {
    for(int i=0;i < fftGrid_numerator->getrsize();i++)
    {
      float numerator   = fftGrid_numerator  ->getNextReal();
      float denominator = fftGrid_denominator->getNextReal();

      if(denominator != 0 ){
        fftGrid_numerator->setNextReal(numerator/denominator);
      }
      else{
        // Do nothing when denominator is zero
        fftGrid_numerator->setNextReal(numerator);
      }
    }
    }
}

void
  GravimetricInversion::ComputeSyntheticGravimetry(FFTGrid * rho, ModelGravityDynamic *& modelGravityDynamic, double level_shift)
{
  LogKit::WriteHeader("Compute Synthetic Gravimetry (and Residuals?)");

  NRLib::Matrix G_fullsize = modelGravityDynamic->GetGMatrixFullSize();
  int N                    = G_fullsize.cols().length();
  int n_obs                = G_fullsize.rows().length();

  NRLib::Vector rho_vec(N);
  VectorizeFFTGrid(rho_vec, rho, false); // Not use padded region

  NRLib::Vector level_shift_vec(n_obs);
  level_shift_vec.initialize(level_shift);

  NRLib::Vector synthetic_data = G_fullsize*rho_vec + level_shift_vec;
  modelGravityDynamic->SetSyntheticData(synthetic_data);

  // Dump synthetic data and full size matrix
  NRLib::WriteVectorToFile("Synthetic_data.txt",  synthetic_data);
}
