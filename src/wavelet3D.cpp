#include <iostream>
#include <fstream>

#include <string.h>
#include <assert.h>
#include <math.h>

#include "fft/include/fftw.h"
#include "fft/include/rfftw.h"
#include "fft/include/fftw-int.h"
#include "fft/include/f77_func.h"

#include "lib/global_def.h"
#include "lib/lib_matr.h"

#include "nrlib/iotools/logkit.hpp"
#include "nrlib/surface/surfaceio.hpp"

#include "src/modelsettings.h"
#include "src/blockedlogs.h"
#include "src/definitions.h"
#include "src/wavelet3D.h"
#include "src/wavelet1D.h"
#include "src/welldata.h"
#include "src/fftgrid.h"
#include "src/simbox.h"
#include "src/model.h"
#include "src/io.h"
#include "src/waveletfilter.h"

Wavelet3D::Wavelet3D(const std::string                          & filterFile,
                     const std::vector<Surface *>               & estimInterval,
                     const NRLib::Grid2D<float>                 & refTimeGradX,
                     const NRLib::Grid2D<float>                 & refTimeGradY,
                     const std::vector<std::vector<double> >    & tGradX,
                     const std::vector<std::vector<double> >    & tGradY,
                     FFTGrid                                    * seisCube,
                     ModelSettings                              * modelSettings,
                     WellData                                  ** wells,
                     Simbox                                     * simBox,
                     float                                      * reflCoef,
                     int                                          angle_index,
                     int                                        & errCode,
                     std::string                                & errText)
  : Wavelet(3),
    filter_(filterFile, errCode, errText)
{
  LogKit::LogFormatted(LogKit::MEDIUM,"  Estimating 3D wavelet pulse from seismic data and (nonfiltered) blocked wells\n");

  theta_      = seisCube->getTheta();;
  norm_       = RMISSING;
  cz_         = 0;
  inFFTorder_ = true;
  isReal_     = true;
  coeff_[0]   = reflCoef[0];
  coeff_[1]   = reflCoef[1];
  coeff_[2]   = reflCoef[2];

  nz_         = simBox->getnz();
  float dx    = static_cast<float>(simBox->getdx());
  float dy    = static_cast<float>(simBox->getdy());
  dz_         = static_cast<float>(simBox->getdz());
  nzp_        = seisCube->getNzp();
  cnzp_       = nzp_/2+1;
  rnzp_       = 2*cnzp_;

  unsigned int nWells    = modelSettings->getNumberOfWells();
  float        v0        = modelSettings->getAverageVelocity();
  float        stretch   = modelSettings->getStretchFactor(angle_index);
  bool         hasHalpha = filter_.hasHalpha();
  
  int nhalfWl         = static_cast<int> (0.5 * modelSettings->getWaveletTaperingL() / dz_);
  int nWl             = 2 * nhalfWl + 1;
  std::vector<std::vector<fftw_real> > wellWavelets(nWells, std::vector<float>(rnzp_, 0.0));
  std::vector<float>                   wellWeight(nWells, 0.0);  
  std::vector<float>                   dzWell(nWells, 0.0);
  for (unsigned int w=0; w<nWells; w++) {
    if (wells[w]->getUseForWaveletEstimation()) {
      LogKit::LogFormatted(LogKit::MEDIUM, "  Well :  %s\n", wells[w]->getWellname().c_str());

      BlockedLogs *bl    = wells[w]->getBlockedLogsOrigThick();  
      const std::vector<int> iPos = bl->getIposVector();
      const std::vector<int> jPos = bl->getJposVector();
      dzWell[w]          = static_cast<float>(simBox->getRelThick(iPos[0],jPos[0])) * dz_;
      
      unsigned int nBlocks = bl->getNumberOfBlocks();
      std::vector<float> zGradX(nBlocks);
      std::vector<float> zGradY(nBlocks);
      std::vector<float> t0GradX(nBlocks);
      std::vector<float> t0GradY(nBlocks);
      for (unsigned int b = 0; b<nBlocks; b++) {
        t0GradX[b]    = refTimeGradX(iPos[b], jPos[b]);
        zGradX[b]     = 0.5f * v0 * 0.001f * (static_cast<float> (tGradX[w][b]) - t0GradX[b]); //0.001f is due to ms/s conversion
        t0GradY[b]    = refTimeGradY(iPos[b], jPos[b]);
        zGradY[b]     = 0.5f * v0 * 0.001f * (static_cast<float> (tGradY[w][b]) - t0GradY[b]);
      }

      std::vector<float> az(nz_); 
      bl->getVerticalTrend(&zGradX[0], &az[0]);
      std::vector<float> bz(nz_);
      bl->getVerticalTrend(&zGradY[0], &bz[0]);
      std::vector<float> at0(nz_); 
      bl->getVerticalTrend(&t0GradX[0], &at0[0]);
      std::vector<float> bt0(nz_);
      bl->getVerticalTrend(&t0GradY[0], &bt0[0]);

      std::vector<bool> hasWellData(nz_);
      findLayersWithData(estimInterval, 
                         bl, 
                         seisCube, 
                         az, 
                         bz, 
                         hasWellData);

      int start, length;
      bl->findContiniousPartOfData(hasWellData, 
                                   nz_, 
                                   start, 
                                   length);
      if (length > nWl) {
        std::vector<fftw_real> cpp(nzp_);
        bl->fillInCpp(coeff_, start, length, &cpp[0], nzp_);
        printVecToFile("cpp", &cpp[0], length);

        std::vector<fftw_real> cppAdj(length, 0.0);
        std::vector<float> Halpha;
        if (hasHalpha)
          Halpha.resize(length,0.0);
        for (int i=start; i < start+length-1; i++) {
          double phi    = findPhi(az[i], bz[i]);
          float r       = sqrt(az[i]*az[i] + bz[i]*bz[i] + 1);
          double psi    = findPsi(r);
          float alpha1  = filter_.getAlpha1(phi, psi);
          cppAdj[i-start]     = static_cast<fftw_real> (cpp[i] * alpha1 * stretch / r);
          if (hasHalpha)
            Halpha[i-start]   = filter_.getHalpha(phi, psi);
        }

        printVecToFile("cpp_dipadjust", &cppAdj[0], length);

        std::vector<float> zLog(nBlocks);
        for (unsigned int b=0; b<nBlocks; b++) {
          double zTop     = simBox->getTop(iPos[b], jPos[b]);
          zLog[b]         = static_cast<float> (zTop + b * simBox->getRelThick(iPos[b], jPos[b]) * dz_);
        }
        std::vector<float> zPosWell(nz_);
        bl->getVerticalTrend(&zLog[0], &zPosWell[0]);

        int nTracesX    = static_cast<int> (modelSettings->getEstRangeX(angle_index) / dx);
        int nTracesY    = static_cast<int> (modelSettings->getEstRangeY(angle_index) / dy);

//        int nMaxPoints = length * (2 * nTracesX + 1) * (2 * nTracesY + 1); 
        std::vector<std::vector<float> > gMat;
        std::vector<float> dVec;
        int nPoints = 0;
        for (int xTr = -nTracesX; xTr <= nTracesX; xTr++) {
          for (int yTr = -nTracesY; yTr <= nTracesY; yTr++) {
            std::vector<float> seisLog(nBlocks);
            bl->getBlockedGrid(&seisCube[0], &seisLog[0], xTr, yTr);
            std::vector<float> seisData(nz_);
            bl->getVerticalTrend(&seisLog[0], &seisData[0]);
            std::vector<float> zLog(nBlocks);
            for (unsigned int b=0; b<nBlocks; b++) {
              int xIndex      = iPos[b] + xTr;
              int yIndex      = jPos[b] + yTr;
              double zTop     = simBox->getTop(xIndex, yIndex);
              zLog[b]         = static_cast<float> (zTop + b * simBox->getRelThick(xIndex, yIndex) * dz_);
            }
            std::vector<float> zPosTrace(nz_);
            bl->getVerticalTrend(&zLog[0], &zPosTrace[0]);
            for (int t=start; t < start+length; t++) {
              if (seisData[t] != RMISSING) {
                dVec.push_back(seisData[t]);
                std::vector<std::vector<float> > lambda(length, std::vector<float>(nWl,0.0)); //NBNB-Frode: B�r denne deklareres utenfor?
                for (int tau = start; tau < start+length; tau++) {
                  //Hva gj�r vi hvis zData[t] er RMISSING. Kan det skje?
                  float at = at0[tau] + 2.0f*az[tau]/v0;
                  float bt = bt0[tau] + 2.0f*bz[tau]/v0;
                  float u = static_cast<float> (zPosTrace[t] - zPosWell[tau] - at*(xTr*dx) - bt*(yTr*dy));
                  if (hasHalpha) {
                    for (int i=0; i<nWl; i++) {
                      float v = u - static_cast<float>((i - nhalfWl)*dz_);
                      float h = Halpha[tau-start];
                      lambda[tau-start][i] = static_cast<float> (h / (PI *(h*h + v*v)));
                    }
                  }
                  else {
                    int tLow  = static_cast<int> (floor(u / dz_));
                    int tHigh = tLow + 1;
                    float lambdaValue = u - static_cast<float> (tLow * dz_);
                    if (u >= 0.0 && tLow <= nhalfWl) { 
                      lambda[tau-start][tLow]   = 1 -lambdaValue;
                      lambda[tau-start][tHigh]  = lambdaValue; 
                    }
                    else if (u < 0.0 && -tHigh <= nhalfWl) {
                      if (tHigh < 0)
                        lambda[tau-start][tHigh+nWl] = lambdaValue;
                      else
                        lambda[tau-start][0] = lambdaValue;
                      if (tLow >= -nhalfWl)
                        lambda[tau-start][tLow+nWl] = 1 - lambdaValue;
                    } // else if
                  } // else
                } // for (tau=start...start+length)
                std::vector<float> gVec(nWl);
                for (int i=0; i<nWl; i++) {
                  gVec[i] = 0.0f;
                  for (int j=0; j<length; j++)
                    gVec[i] += cppAdj[j] * lambda[j][i];
                }
                gMat.push_back(gVec);
                nPoints++;
              } //if (seisData[t] != RMISSING)
            }
          }
        }

        printMatToFile("design_matrix", gMat, nPoints, nWl);

        double **gTrg = new double *[nWl];
        for (int i=0; i<nWl; i++) {
          gTrg[i]     = new double[nWl];
          for (int j=0; j<nWl; j++) {
            gTrg[i][j] = 0.0;
            for (int k=0; k<nPoints; k++)
              gTrg[i][j] += gMat[k][i]*gMat[k][j];
          }
        }

        printVecToFile("seismic", &dVec[0], nPoints);

        double *gTrd = new double[nWl];
        for (int i=0; i<nWl; i++) {
          gTrd[i] = 0.0;
          for (int j=0; j<nPoints; j++)
            gTrd[i] += gMat[j][i] * dVec[j];
        }

        lib_matrCholR(nWl, gTrg);
        lib_matrAxeqbR(nWl, gTrg, gTrd);
        for (int i=0; i<nWl; i++)
          delete [] gTrg[i];
        delete [] gTrg;
 
        wellWavelets[w][0] = static_cast<fftw_real> (gTrd[0]);
        for (int i=0; i<nhalfWl; i++) {
          wellWavelets[w][i]      = static_cast<fftw_real> (gTrd[i]);
          wellWavelets[w][nzp_-1] = static_cast<fftw_real> (gTrd[nWl-i]);
        }
        for (int i=nhalfWl+1; i<nzp_-nhalfWl; i++)
          wellWavelets[w][i] = 0.0f;
        for (int i=nzp_; i<rnzp_; i++)
          wellWavelets[w][i] = RMISSING;
        delete [] gTrd;

        double s2 = 0.0;
        for (int i=0; i<nPoints; i++) {
          double prod = 0.0;
          for (int j=0; j<nWl; j++)
            prod += static_cast<double> (gMat[i][j]*wellWavelets[w][j]);
          double residual = prod - dVec[i];
          s2 += residual * residual;
        }
        wellWeight[w] = static_cast<float> (1/s2);
      } //if (length > nWl)
      else {
        LogKit::LogFormatted(LogKit::MEDIUM,"     No enough data for 3D wavelet estimation in well %s\n", wells[w]->getWellname().c_str());
      }
    } // if(wells->getUseForEstimation)
  } // for (w=0...nWells) 

  rAmp_ = averageWavelets(wellWavelets, nWells, nzp_, wellWeight, dzWell, dz_);
  cAmp_ = reinterpret_cast<fftw_complex*>(rAmp_);
  waveletLength_ = findWaveletLength(modelSettings->getMinRelWaveletAmp());
  LogKit::LogFormatted(LogKit::LOW,"  Estimated wavelet length:  %.1fms\n",waveletLength_);

  if( ModelSettings::getDebugLevel() > 0 )
    writeWaveletToFile("estimated_wavelet", 1.0f);

  double norm2=0.0;
  for(int i=0; i < nzp_; i++ )
      norm2 += static_cast<double> (rAmp_[i]*rAmp_[i]);
  norm_= static_cast<float>(sqrt(norm2));

  fftw_real * trueAmp = rAmp_;
  rAmp_               = static_cast<fftw_real*>(fftw_malloc(rnzp_*sizeof(fftw_real)));
  cAmp_               = reinterpret_cast<fftw_complex *>(rAmp_);

  for(unsigned int w=0; w<nWells; w++) {
    for(int i=0; i<nzp_; i++)
      rAmp_[i] = wellWavelets[w][i];
    std::string fileName = "Wavelet"; 
    std::string wellname(wells[w]->getWellname());
    NRLib::Substitute(wellname,"/","_");
    NRLib::Substitute(wellname," ","_");

    fileName += "_"+wellname; 
    writeWaveletToFile(fileName, 1.0f);
  }
  fftw_free(rAmp_);
  rAmp_ = trueAmp;
  cAmp_ = reinterpret_cast<fftw_complex *>(rAmp_);
}


Wavelet3D::Wavelet3D(const std::string & fileName, 
            int                 fileFormat, 
            ModelSettings     * modelSettings, 
            float             * reflCoef,
            float               theta,
            int               & errCode, 
            std::string       & errText,
            const std::string & filterFile)
  : Wavelet(fileName, fileFormat, modelSettings, reflCoef, theta, 3, errCode, errText),
    filter_(filterFile, errCode, errText)
{
}

Wavelet3D::Wavelet3D(Wavelet * wavelet)
  : Wavelet(3, wavelet)
{
}

Wavelet3D::~Wavelet3D()
{
}

void
Wavelet3D::findLayersWithData(const std::vector<Surface *> & estimInterval,
                              BlockedLogs                  * bl,
                              FFTGrid                      * seisCube,
                              const std::vector<float>     & az,
                              const std::vector<float>     & bz,
                              std::vector<bool>            & hasWellData) const
{
  std::vector<float> seisLog(bl->getNumberOfBlocks());
  bl->getBlockedGrid(seisCube, &seisLog[0]);
  std::vector<float> seisData(nz_);
  bl->getVerticalTrend(&seisLog[0], &seisData[0]);

  std::vector<float> alpha(nz_);
  std::vector<float> beta(nz_);
  std::vector<float> rho(nz_);
  bl->getVerticalTrend(bl->getAlpha(), &alpha[0]);
  bl->getVerticalTrend(bl->getBeta(), &beta[0]);
  bl->getVerticalTrend(bl->getRho(), &rho[0]);

  for (int k=0; k<nz_; k++) 
    hasWellData[k] = (alpha[k] != RMISSING && beta[k] != RMISSING && rho[k] != RMISSING && az[k] != RMISSING && bz[k] != RMISSING && seisData[k] != RMISSING);

  //Check that data are within wavelet estimation interval
  if (estimInterval.size() > 0) {
    const std::vector<double> xPos = bl->getXposVector();
    const std::vector<double> yPos = bl->getYposVector();
    const std::vector<double> zPos = bl->getZposVector();
    for (int k=0; k<nz_; k++) {
      const double zTop  = estimInterval[0]->GetZ(xPos[k],yPos[k]);
      const double zBase = estimInterval[1]->GetZ(xPos[k],yPos[k]);
      if ((zPos[k]-0.5*dz_) < zTop || (zPos[k]+0.5*dz_) > zBase)
        hasWellData[k] = false;
    }
  }
}

double 
Wavelet3D::findPhi(float a, float b) const
//Return value should be between 0 and 360
{
  double phi;
  double epsilon = 0.001;
  if (a > epsilon && b >= 0.0) //1. quadrant 
    phi = atan(b/a);
  else if (a > epsilon && b < 0.0) //4. quadrant
    phi = 2*PI + atan(b/a);
  else if (a < - epsilon && b >= 0.0) //2. quadrant
    phi = PI + atan(b/a);
  else if (a < - epsilon && b < 0.0) //3. quadrant
    phi = PI + atan(b/a);
  else if (b  >= 0.0) //kx very small
    phi = 0.5 * PI;
  else //kx very small
    phi = 1.5 * PI;
  phi = phi * 180 / PI;
  return(phi);
}

double 
Wavelet3D::findPsi(float r) const
//Return value should be between 0 and 90
{
  double psi    = acos(1.0 / r);
  psi = psi * 180 / PI;
  return(psi);
}

fftw_complex 
Wavelet3D::findWLvalue(float omega) const
{
  int lowindex = static_cast<int> (omega / dz_);
  fftw_complex c_low, c_high;
  if (lowindex >= nz_) {
    c_low.re = 0.0; 
    c_low.im = 0.0;
    c_high.re = 0.0;
    c_high.im = 0.0;
  }
  else if (lowindex == nz_-1) {
    c_high.re = 0.0;
    c_high.im = 0.0;
    c_low = getCAmp(lowindex);
  }
  else {
    c_low = getCAmp(lowindex);
    c_high = getCAmp(lowindex + 1);
  }
  float fac = omega - lowindex * dz_;
  fftw_complex cValue;
  cValue.re = (1-fac) * c_low.re + fac * c_high.re;
  cValue.im = (1-fac) * c_low.im + fac * c_high.im;

  return cValue;
}
/*
void
Wavelet3D::printVecToFile(const std::string & fileName, 
                          const std::vector<float> & vec, 
                          int n) const
{
  if( ModelSettings::getDebugLevel() > 0) { 
    std::string fName = fileName + IO::SuffixGeneralData();
    fName = IO::makeFullFileName(IO::PathToWavelets(), fName);
    std::ofstream file;
    NRLib::OpenWrite(file,fName);
    for(int i=0;i<n;i++)
      file << vec[i] << "\n";
    file.close();
  }  
}
*/
void
Wavelet3D::printMatToFile(const std::string                       & fileName, 
                          const std::vector<std::vector<float> >  & mat, 
                          int                                       n,
                          int                                       m) const
{
  if( ModelSettings::getDebugLevel() > 0) { 
    std::string fName = fileName + IO::SuffixGeneralData();
    fName = IO::makeFullFileName(IO::PathToWavelets(), fName);
    std::ofstream file;
    NRLib::OpenWrite(file,fName);
    for(int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++)
        file << mat[i][j];
      file << "\n";
    }
    file.close();
  }  
}
