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
#include "lib/lib_misc.h"
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

Wavelet3D::Wavelet3D(const std::string   & filterFile,
                     const std::string   & refTimeFile,
                     ModelSettings       * /*modelSettings*/,
                     WellData           ** /*wells*/,
                     int                   /*angle_index*/,
                     Simbox              * simBox,
                     float                 theta,
                     int                 & errCode,
                     char                * errText)
  : Wavelet(3),
    wavelet1D_(NULL),
    filter_(filterFile, errCode, errText)
{
//  float v0 = modelSettings->getAverageVelocity();
  theta_ = theta;
  norm_ = RMISSING;

  NRLib::RegularSurfaceRotated<double> t0surface = NRLib::ReadSgriSurf(refTimeFile);

  findTimeGradientSurface(t0surface,
                          simBox,
                          errCode,
                          errText);
}


Wavelet3D::Wavelet3D(Wavelet1D           * wavelet1D,
                     const std::string   & filterFile,
                     ModelSettings       * modelSettings,
                     int                   angle_index,
                     Simbox              * simBox,
                     float                 theta,
                     int                 & errCode,
                     char                * errText)
  : Wavelet(3),
    wavelet1D_(wavelet1D),
    filter_(filterFile, errCode, errText)
{
  float v0 = modelSettings->getAverageVelocity();
  nx_ = simBox->getnx();
  ny_ = simBox->getny();
  nz_ = simBox->getnz();
  dx_ = static_cast<float>(simBox->getdx());
  dy_ = static_cast<float>(simBox->getdy());
  dz_ = static_cast<float>(simBox->getdz() * 0.5f * v0 * 0.001f);

  double xPadFac = modelSettings->getXPadFac();
  nxp_   =  findClosestFactorableNumber( static_cast<int>(ceil( static_cast<double>(nx_)*(1.0+xPadFac) )) );
  double yPadFac = modelSettings->getYPadFac();
  nyp_   =  findClosestFactorableNumber( static_cast<int>(ceil( static_cast<double>(ny_)*(1.0+yPadFac) )) );
  double zPadFac = modelSettings->getZPadFac();
  nzp_   =  findClosestFactorableNumber( static_cast<int>(ceil( static_cast<double>(nz_)*(1.0+zPadFac) )) );
  
  theta_ = theta;
  norm_ = RMISSING;

  wavelet1D_->resample(static_cast<float>(simBox->getdz()), 
                      nz_, 
                      static_cast<float> (zPadFac), 
                      theta);
  wavelet1D_->fft1DInPlace();
    
  ampCube_ = FFTGrid(nx_, ny_, nz_, nxp_, nyp_, nzp_);
  ampCube_.createComplexGrid();
  ampCube_.setType(FFTGrid::COVARIANCE);
  ampCube_.setAccessMode(FFTGrid::RANDOMACCESS);

  int cnxp = nxp_/2+1;
  int i, j, k;
  float kx, ky, kz;
  float radius, alpha1, hAlpha, alpha2, omega;
  double phi, psi;
  fftw_complex cValue;
  
  float stretch = modelSettings->getStretchFactor(angle_index);
  float minus2pi = static_cast<float> (-2.0 * PI);

  for (k=0; k<=nzp_/2; k++) {
    kz = static_cast<float> (k / dz_);
    for (j=0; j<=nyp_/2; j++) {
      ky = static_cast<float> (j / dy_);
      for (i=0; i<cnxp; i++) {
        kx = static_cast<float> (i / dx_);
        radius = sqrt(kx*kx + ky*ky + kz*kz);
        phi = findPhi(kx, ky);
        psi = findPsi(radius, kz);
        alpha1 = static_cast<float> (filter_.getAlpha1(phi, psi));
        hAlpha = static_cast<float> (filter_.getHalpha(phi, psi));
        omega = (0.5f * v0 * radius) / stretch;
        cValue = findWLvalue(wavelet1D_, omega);
        alpha2 = exp(minus2pi * omega * hAlpha);
        cValue.re *= static_cast<fftw_real> (alpha1 * alpha2);
        setCAmp(cValue,k,j,i);
      }
    }
    for (j=(nyp_/2)+1; j<nyp_; j++) {
      ky = static_cast<float> ((j-nyp_) / dy_);
      for (i=0; i<cnxp; i++) {
        kx = static_cast<float> (i / dx_);
        radius = sqrt(kx*kx + ky*ky + kz*kz);
        phi = findPhi(kx, ky);
        psi = findPsi(radius, kz);
        alpha1 = static_cast<float> (filter_.getAlpha1(phi, psi));
        hAlpha = static_cast<float> (filter_.getHalpha(phi, psi));
        omega = (0.5f * v0 * radius) / stretch;
        cValue = findWLvalue(wavelet1D_, omega);
        alpha2 = exp(minus2pi * omega * hAlpha);
        cValue.re *= static_cast<fftw_real> (alpha1 * alpha2);
        setCAmp(cValue,k,j,i);
      }
    }
  }
  for (k=(nzp_/2)+1; k<nzp_; k++) {
    kz = static_cast<float> ((nzp_-k) / dz_);
    for (j=0; j<=nyp_/2; j++) {
      ky = static_cast<float> (j / dy_);
      for (i=0; i<cnxp; i++) {
        kx = static_cast<float> (i / dx_);
        radius = sqrt(kx*kx + ky*ky + kz*kz);
        phi = findPhi(kx, ky);
        psi = findPsi(radius, kz);
        alpha1 = static_cast<float> (filter_.getAlpha1(phi, psi));
        hAlpha = static_cast<float> (filter_.getHalpha(phi, psi));
        omega = (0.5f * v0 * radius) / stretch;
        cValue = findWLvalue(wavelet1D_, omega);
        alpha2 = exp(minus2pi * omega * hAlpha);
        cValue.re *= static_cast<fftw_real> (alpha1 * alpha2);
        setCAmp(cValue,k,j,i);
      }
    }
    for (j=(nyp_/2)+1; j<nyp_; j++) {
      ky = static_cast<float> ((nyp_-j) / dy_);
      for (i=0; i<cnxp; i++) {
        kx = static_cast<float> (i / dx_);
        radius = sqrt(kx*kx + ky*ky + kz*kz);
        phi = findPhi(kx, ky);
        psi = findPsi(radius, kz);
        alpha1 = static_cast<float> (filter_.getAlpha1(phi, psi));
        hAlpha = static_cast<float> (filter_.getHalpha(phi, psi));
        omega = (0.5f * v0 * radius) / stretch;
        cValue = findWLvalue(wavelet1D_, omega);
        alpha2 = exp(minus2pi * omega * hAlpha);
        cValue.re *= static_cast<fftw_real> (alpha1 * alpha2);
        setCAmp(cValue,k,j,i);
      }
    }
  }

  FFTGrid *shiftAmp = new FFTGrid(nx_, ny_, nz_, nx_, ny_, nz_);
  shiftAmp->fillInConstant(0.0);
  shiftAmp->setType(FFTGrid::DATA);
  shiftAmp->setAccessMode(FFTGrid::RANDOMACCESS);

  shiftFFTGrid(shiftAmp);

  std::ofstream headerFile;
  NRLib::OpenWrite(headerFile, "WL_as_shiftedFFTGrid.Sgrh");

  headerFile << "NORSAR General Grid Format v1.0\n";
  headerFile << "3\n";
  headerFile << "kx (1/m)\n";
  headerFile << "ky (1/m)\n";
  headerFile << "kz (1/m)\n";
  headerFile << "FFT-grid\n";
  headerFile << "1\n";
  headerFile << "3D-wavelet" << std::endl;
  headerFile << "1 1 1\n";
  headerFile << nx_ << " " << ny_ << " " << nz_ << std::endl;
  headerFile << std::setprecision(10);
  headerFile << 1/dx_ << " " << 1/dy_ << " " << 1/dz_ << std::endl;
//  double x0 = simbox->getx0() + 0.5 * simbox->getdx();
//  double y0 = simbox->gety0() + 0.5 * simbox->getdy();
//  double z0 = zMin + 0.5 * dz;
  headerFile << "0.0 0.0 0.0\n";
  headerFile << "0.0\n";
  headerFile << RMISSING << std::endl;
  headerFile << "WL_as_shiftedFFTGrid.Sgri\n";
  headerFile.close();

  std::ofstream binFile;
  NRLib::OpenWrite(binFile, "WL_as_shiftedFFTGrid.Sgri" , std::ios::out | std::ios::binary);
  float value;
  for (k=0; k<nz_; k++)
    for (j=0; j<ny_; j++)
      for (i=0; i<nx_; i++) {
          value = shiftAmp->getRealValue(i,j,k);
#ifndef BIGENDIAN
        NRLib::WriteBinaryFloat(binFile, value);
#else
        NRLib::WriteBinaryFloat(binFile, value, END_LITTLE_ENDIAN);
#endif
      }
  binFile.close();
}


Wavelet3D::Wavelet3D(Wavelet * wavelet, int difftype)
  : Wavelet(wavelet, 3)
{
  assert(wavelet->getDim() == 3);
  assert(wavelet->getIsReal());
  nx_ = wavelet->getNx();
  ny_ = wavelet->getNy();
  nxp_ = wavelet->getNxp();
  nyp_ = wavelet->getNyp();
  dx_ = wavelet->getDx();
  dy_ = wavelet->getDy();
  ampCube_ = FFTGrid(nx_, ny_, nz_, nxp_, nyp_, nzp_);
  ampCube_.createRealGrid();
  ampCube_.setType(FFTGrid::COVARIANCE);
  ampCube_.setAccessMode(FFTGrid::RANDOMACCESS);
  float rValue;
  for (int i=0; i<nxp_; i++) {
    for (int j=0; j<nyp_; j++) {
      for (int k=0; k<nzp_; k++) {
        if (difftype == FIRSTORDERFORWARDDIFF) {
          if (k == nzp_-1) 
            rValue = wavelet->getRAmp(0,j,i) - wavelet->getRAmp(k,j,i);
          else
            rValue = wavelet->getRAmp(k+1,j,i) - wavelet->getRAmp(k,j,i);
        }
        else { //(difftype == FIRSTORDERBACKWARDDIFF)
          if (k == 0)
            rValue = wavelet->getRAmp(k,j,i) - wavelet->getRAmp(nzp_-1,j,i);
          else
            rValue = wavelet->getRAmp(k,j,i) -wavelet->getRAmp(k-1,j,i);
        }
        ampCube_.setRealValue(i,j,k,rValue,true);
      }
    }
  }
}

Wavelet3D::Wavelet3D(Wavelet * wavelet)
  : Wavelet(wavelet, 3)
{
  assert(wavelet->getDim() == 3);
  assert(wavelet->getIsReal());
  nx_         = wavelet->getNx();
  ny_         = wavelet->getNy();
  nxp_        = wavelet->getNxp();
  nyp_        = wavelet->getNyp();
  dx_         = wavelet->getDx();
  dy_         = wavelet->getDy();

  ampCube_ = FFTGrid(nx_, ny_, nz_, nxp_, nyp_, nzp_);
  ampCube_.createRealGrid();
  ampCube_.setType(FFTGrid::COVARIANCE);
  ampCube_.setAccessMode(FFTGrid::RANDOMACCESS);
  for (int k=0; k<nzp_; k++) {
    for (int j=0; j<nyp_; j++) {
      for (int i=0; i<nxp_; i++) {
        float rvalue = wavelet->getRAmp(k,j,i);
        ampCube_.setRealValue(i,j,k,rvalue,true);
      }
    }
  }
}

Wavelet3D::Wavelet3D(Wavelet3D *wavelet)
  : Wavelet(wavelet, 3),
    wavelet1D_(wavelet->getWavelet1D())
{
  assert(wavelet->getDim() == 3);
  assert(wavelet->getIsReal());
  nx_         = wavelet->getNx();
  ny_         = wavelet->getNy();
  nxp_        = wavelet->getNxp();
  nyp_        = wavelet->getNyp();
  dx_         = wavelet->getDx();
  dy_         = wavelet->getDy();

  ampCube_ = FFTGrid(nx_, ny_, nz_, nxp_, nyp_, nzp_);
  ampCube_.createRealGrid();
  ampCube_.setType(FFTGrid::COVARIANCE);
  ampCube_.setAccessMode(FFTGrid::RANDOMACCESS);
  for (int k=0; k<nzp_; k++) {
    for (int j=0; j<nyp_; j++) {
      for (int i=0; i<nxp_; i++) {
        float rvalue = wavelet->getRAmp(k,j,i);
        ampCube_.setRealValue(i,j,k,rvalue,true);
      }
    }
  }
  filter_     = wavelet->getFilter();
}

double Wavelet3D::findPhi(float kx, float ky) const
//Return value should be between 0 and 2*PI
{
  double phi;
  double epsilon = 0.000001;
  if (kx > epsilon && ky >= 0.0) //1. quadrant 
    phi = atan(ky/kx);
  else if (kx > epsilon && ky < 0.0) //4. quadrant
    phi = 2*PI + atan(ky/kx);
  else if (kx < - epsilon && ky >= 0.0) //2. quadrant
    phi = PI + atan(ky/kx);
  else if (kx < - epsilon && ky < 0.0) //3. quadrant
    phi = PI + atan(ky/kx);
  else if (ky  >= 0.0) //kx very small
    phi = 0.5 * PI;
  else //kx very small
    phi = 1.5 * PI;

  return(phi);
}

double Wavelet3D::findPsi(float radius, float kz) const
//Return value should be between 0 and 0.5*PI
{
  double epsilon = 0.000001;
  double psi = 0.0;
  if (kz < 0.0)
    kz = -kz;
  if (radius > epsilon)
    psi = acos(kz/radius);

  return(psi);
}

fftw_complex Wavelet3D::findWLvalue(Wavelet1D       * wavelet1d,
                                    float             omega) const
{
  int lowindex = static_cast<int> (omega / wavelet1d->getDz());
  fftw_complex c_low, c_high;
  if (lowindex >= wavelet1d->getNz()) {
    c_low.re = 0.0; 
    c_low.im = 0.0;
    c_high.re = 0.0;
    c_high.im = 0.0;
  }
  else if (lowindex == wavelet1d->getNz()-1) {
    c_high.re = 0.0;
    c_high.im = 0.0;
    c_low = wavelet1d->getCAmp(lowindex);
  }
  else {
    c_low = wavelet1d->getCAmp(lowindex);
    c_high = wavelet1d->getCAmp(lowindex + 1);
  }
  float fac = omega - lowindex * wavelet1d->getDz();
  fftw_complex cValue;
  cValue.re = (1-fac) * c_low.re + fac * c_high.re;
  cValue.im = (1-fac) * c_low.im + fac * c_high.im;

  return cValue;
}


bool 
Wavelet3D::findTimeGradientSurface(const NRLib::RegularSurfaceRotated<double>   & rot_surface,
                                   Simbox                                       * simbox,
                                   int                                          & errCode,
                                   char                                         * errText)
{
  double x, y;
  bool inside = true;
  unsigned int nx = static_cast<unsigned int> (simbox->getnx());
  unsigned int ny = static_cast<unsigned int> (simbox->getny());
  double dx = simbox->getdx();
  double dy = simbox->getdy();

  simbox->getXYCoord(0,0,x,y);
  if (rot_surface.IsInsideSurface(x,y)) {
    simbox->getXYCoord(0,ny-1,x,y);
    if (rot_surface.IsInsideSurface(x,y)) {
      simbox->getXYCoord(nx-1,0,x,y);
      if (rot_surface.IsInsideSurface(x,y)) {
        simbox->getXYCoord(nx-1,ny-1,x,y);
        if (!rot_surface.IsInsideSurface(x,y))
          inside = false;
      }
      else
        inside = false;
    }
    else
      inside = false;
  }
  else
    inside = false;

  if (inside) {
    x_gradient_.Resize(nx, ny, RMISSING);
    y_gradient_.Resize(nx, ny, RMISSING);
    for (unsigned int i = nx-1; i >= 0; i++) {
      for (unsigned int j = ny-1; j >= 0; j++) {
        simbox->getXYCoord(i,j,x,y);
        double z_high = rot_surface.GetZInside(x,y);
        simbox->getXYCoord(i,j-1,x,y); //XYCoord is ok even if j = -1, but point is outside simbox
        double z_low = rot_surface.GetZInside(x,y);
        if (!rot_surface.IsMissing(z_low))
          y_gradient_(i,j) = (z_high - z_low) / dy;
        else
          y_gradient_(i,j) = y_gradient_(i,j+1);

        simbox->getXYCoord(i-1,j,x,y); //XYCoord is ok even if j = -1, but point is outside simbox
        z_low = rot_surface.GetZInside(x,y);
        if (!rot_surface.IsMissing(z_low))
          x_gradient_(i,j) = (z_high - z_low) / dx;
        else
          x_gradient_(i,j) = x_gradient_(i+1,j);
      }
    }
  }
  else {
    sprintf(errText, "%sSimbox is not completely inside reference time surface in (x,y).\n", errText);
    errCode = 1;
  }

  return(inside);
}

void           
Wavelet3D::fft1DInPlace()
{
  ampCube_.fftInPlace();
  isReal_ = false;
}

void
Wavelet3D::invFFT1DInPlace()
{
  ampCube_.invFFTInPlace();
  isReal_ = true;
}

bool           
Wavelet3D::consistentSize(int nzp, int nyp, int nxp) const 
{ 
  bool ok = true;
  if (nzp!=nzp_) {
    printf("nzp=%d  nzp_wavelet3D=%d\n",nzp,nzp_);
    ok = false;
  }
  if (nyp != nyp_) {
    printf("nyp=%d  nyp_wavelet3D=%d\n",nyp,nyp_);
    ok = false;
  }
  if (nxp != nxp_) {
    printf("nxp=%d  nxp_wavelet3D=%d\n",nxp,nxp_);
    ok = false;
  }
  return (ok);
}

fftw_complex   
Wavelet3D::getCAmp(int k, int j, int i) const
{
  return(ampCube_.getComplexValue(i,j,k,true));
}

fftw_real      
Wavelet3D::getRAmp(int k, int j, int i)
{
  return(ampCube_.getRealValue(i,j,k,true));
}

fftw_complex   
Wavelet3D::getCAmp(int k, float, int j, int i) const
{
  return(ampCube_.getComplexValue(i,j,k,true));
}

void           
Wavelet3D::setRAmp(float value, int k, int j, int i)
{
  ampCube_.setRealValue(i,j,k,value);
}

void
Wavelet3D::setCAmp(fftw_complex value, int k, int j, int i)
{
  ampCube_.setComplexValue(i, j ,k, value, true);
}

void
Wavelet3D::scale(float scale)
{
  Wavelet::scale(scale);

  float rAmp;
  for(int i=0; i < nxp_ ; i++) {
    for (int j=0; j < nyp_; j++) {
      for (int k=0; k < nzp_; k++) {
        rAmp = getRAmp(k,j,i);
        if (rAmp != RMISSING) {
          rAmp *= scale;
          setRAmp(rAmp,k,j,i);
        }
      }
    }
  }
}

void
Wavelet3D::multiplyByR(float p)
{
  assert(!getIsReal());
  float scale = static_cast<float> (1.0/(nxp_*nyp_*nzp_));
  int cnxp = nxp_/2+1;
  int i, j, k;
  float kx, ky, kz;
  float radius;
  fftw_complex cValue;

  for (k=0; k<=nzp_/2; k++) {
    kz = static_cast<float> (k);
    for (j=0; j<=nyp_/2; j++) {
      ky = static_cast<float> (j);
      for (i=0; i<cnxp; i++) {
        kx = static_cast<float> (i);
        radius = sqrt(kx*kx + ky*ky + kz*kz);
        cValue = getCAmp(k,j,i);
        cValue.re *= scale * pow(radius, p);
        cValue.im *= scale * pow(radius, p);
        setCAmp(cValue,k,j,i);
      }
    }
    for (j=(nyp_/2)+1; j<nyp_; j++) {
      ky = static_cast<float> (nyp_-j);
      for (i=0; i<cnxp; i++) {
        kx = static_cast<float> (i);
        radius = sqrt(kx*kx + ky*ky + kz*kz);
        cValue = getCAmp(k,j,i);
        cValue.re *= scale * pow(radius, p);
        cValue.im *= scale * pow(radius, p);
        setCAmp(cValue,k,j,i);
      }
    }
  }
  for (k=(nzp_/2)+1; k<nzp_; k++) {
    kz = static_cast<float> (nzp_-k);
    for (j=0; j<=nyp_/2; j++) {
      ky = static_cast<float> (j);
      for (i=0; i<cnxp; i++) {
        kx = static_cast<float> (i);
        radius = sqrt(kx*kx + ky*ky + kz*kz);
        cValue = getCAmp(k,j,i);
        cValue.re *= scale * pow(radius, p);
        cValue.im *= scale * pow(radius, p);
        setCAmp(cValue,k,j,i);
      }
    }
    for (j=(nyp_/2)+1; j<nyp_; j++) {
      ky = static_cast<float> (nyp_-j);
      for (i=0; i<cnxp; i++) {
        kx = static_cast<float> (i);
        radius = sqrt(kx*kx + ky*ky + kz*kz);
        cValue = getCAmp(k,j,i);
        cValue.re *= scale * pow(radius, p);
        cValue.im *= scale * pow(radius, p);
        setCAmp(cValue,k,j,i);
      }
    }
  }
}



void
Wavelet3D::printToFile(const std::string & fileName, bool overrideDebug) 
{
  if(overrideDebug == true || ModelSettings::getDebugLevel() > 0) {
    std::string fName = fileName + IO::SuffixGeneralData();
    fName = IO::makeFullFileName(IO::PathToWavelets(), fName);
    std::ofstream file;
    NRLib::OpenWrite(file, fName);
    LogKit::LogFormatted(LogKit::LOW,"\nWriting STORM ascii file "+fName+"...");
    for(int k=0;k<nzp_;k++)
      for(int j=0;j<nyp_;j++) {
        for(int i=0;i<nxp_;i++) {
          file << getRAmp(k,j,i);
        }
        file << "\n";
      }
    file << "0\n";
    file.close();
  }
}

void
Wavelet3D::writeWaveletToFile(const std::string & fileName, float, Simbox *simbox)
{
  LogKit::LogFormatted(LogKit::MEDIUM,"  Writing 3D-Wavelet to file. \n");
  ampCube_.writeFile(fileName, IO::PathToWavelets(), simbox, "", false);
}

void
Wavelet3D::shiftFFTGrid(FFTGrid *shiftAmp)
{
  int cx = nx_/2;
  int cy = ny_/2;
  int cz = nz_/2;
  int xOdd=0 , yOdd=0, zOdd=0;
  if (nx_/2 != (nx_+1)/2)
    xOdd=1;
  if (ny_/2 != (ny_+1)/2)
    yOdd=1;
  if (nz_/2 != (nz_+1)/2)
    zOdd=1;

  int i,j,k;
  float value;
  for (i=0; i<cx+xOdd; i++) {
    for (j=0; j<cy+yOdd; j++) {
      for (k=0; k<cz+zOdd; k++) {
        value = ampCube_.getRealValue(i,j,k,true);
        shiftAmp->setRealValue(i+cx,j+cy,k+cz,value);
      }
      for (k=nzp_-cz; k<nzp_; k++) {
        value = ampCube_.getRealValue(i,j,k,true);
        shiftAmp->setRealValue(i+cx,j+cy,k-(nzp_-cz),value);
      }
    }
    for (j=nyp_-cy; j<nyp_; j++) {
      for (k=0; k<cz; k++) {
        value = ampCube_.getRealValue(i,j,k,true);
        shiftAmp->setRealValue(i+cx,j-(nyp_-cy),k+cz,value);
      }
      for (k=nzp_-cz; k<nzp_; k++) {
        value = ampCube_.getRealValue(i,j,k,true);
        shiftAmp->setRealValue(i+cx,j-(nyp_-cy),k-(nzp_-cz),value);
      }
    }
  }
  for (i=nxp_-cx; i<nxp_; i++) {
    for (j=0; j<cy; j++) {
      for (k=0; k<cz; k++) {
        value = ampCube_.getRealValue(i,j,k,true);
        shiftAmp->setRealValue(i-(nxp_-cx),j+cy,k+cz,value);
      }
      for (k=nzp_-cz; k<nzp_; k++) {
        value = ampCube_.getRealValue(i,j,k,true);
        shiftAmp->setRealValue(i-(nxp_-cx),j+cy,k-(nzp_-cz),value);
      }
    }
    for (j=nyp_-cy; j<nyp_; j++) {
      for (k=0; k<cz; k++) {
        value = ampCube_.getRealValue(i,j,k,true);
        shiftAmp->setRealValue(i-(nxp_-cx),j-(nyp_-cy),k+cz,value);
      }
      for (k=nzp_-cz; k<nzp_; k++) {
        value = ampCube_.getRealValue(i,j,k,true);
        shiftAmp->setRealValue(i-(nxp_-cx),j-(nyp_-cy),k-(nzp_-cz),value);
      }
    }
  }
}
