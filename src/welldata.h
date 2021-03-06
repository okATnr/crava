/***************************************************************************
*      Copyright (C) 2008 by Norwegian Computing Center and Statoil        *
***************************************************************************/

#ifndef WELLDATA_H
#define WELLDATA_H

#include "nrlib/stormgrid/stormcontgrid.hpp"

#include "src/blockedlogs.h"
#include "src/simbox.h"
#include <string.h>
#include <stdio.h>

class ModelSettings;

namespace NRLib {
  class Well;
}

class WellData
{
public:
  WellData(const std::string & wellFileName,
           ModelSettings     * modelSettings,
           int                 iWell);
  ~WellData(void);

  const float       * getAlpha(int &nData) const;
  const float       * getBeta(int &nData) const;
  const float       * getRho(int &nData) const;
  const int         * getFacies(int &nData) const;
  const float       * getPorosity(int &nData) const;
  const float       * getAlphaBackgroundResolution(int &nData) const;
  const float       * getBetaBackgroundResolution(int &nData) const;
  const float       * getRhoBackgroundResolution(int &nData) const;
  const float       * getAlphaSeismicResolution(int &nData) const;
  const float       * getBetaSeismicResolution(int &nData) const;
  const float       * getRhoSeismicResolution(int &nData) const;
  const double      * getXpos(int &nData) const;
  const double      * getYpos(int &nData) const;
  const double      * getZpos(int &nData) const;
  const double      * getMD(int &nData) const;
  const std::string   getWellname(void)                  const { return wellname_                      ;}
  float               getMeanVsVp(void)                  const { return meanVsVp_                      ;}
  int                 getNumberOfVsVpSamples(void)       const { return nVsVp_                         ;}
  bool                hasSyntheticVsLog(void)            const { return(realVsLog_ == 0)               ;}
  bool                isDeviated(void)                   const { return isDeviated_                    ;}
  bool                getUseForFaciesProbabilities(void) const { return(useForFaciesProbabilities_ > 0);}
  bool                getUseForFiltering(void)           const { return(useForFiltering_ > 0)          ;}
  bool                getUseForWaveletEstimation(void)   const { return(useForWaveletEstimation_ > 0)  ;}
  bool                getUseForBackgroundTrend(void)     const { return(useForBackgroundTrend_ > 0)    ;}
  bool                isFaciesLogDefined(void) const;
  int                 getNFacies(void)                   const { return nFacies_                   ;}
  int               * getFaciesNr(void)                  const { return faciesNr_                  ;}
  int                 getFaciesNr(int i)                 const { return faciesNr_[i]               ;}
  std::string         getFaciesName(int i)               const { return faciesNames_[i]            ;}
  void                getMinMaxFnr(int &min, int &max) const;
  void                findMeanVsVp(void);

  BlockedLogs       * getBlockedLogsOrigThick(void)      const { return blockedLogsOrigThick_  ;}
  BlockedLogs       * getBlockedLogsConstThick(void)     const { return blockedLogsConstThick_ ;}
  BlockedLogs       * getBlockedLogsExtendedBG(void)     const { return blockedLogsExtendedBG_ ;}

  void                deleteBlockedLogsOrigThick(void)           { delete blockedLogsOrigThick_ ;}
  void                setBlockedLogsOrigThick(BlockedLogs * bl)  { blockedLogsOrigThick_  = bl  ;}
  void                setBlockedLogsConstThick(BlockedLogs * bl) { blockedLogsConstThick_ = bl  ;}
  void                setBlockedLogsExtendedBG(BlockedLogs * bl) { blockedLogsExtendedBG_ = bl  ;}

  int                 getNd(void) const;
  int                 checkError(std::string & errText);
  int                 checkSimbox(Simbox *simbox);
  int                 checkVolume(NRLib::Volume & volume) const;
  bool                removeDuplicateLogEntries(const Simbox * simbox, int & nMerges);
  void                setWrongLogEntriesUndefined(int & count_alpha, int & count_beta, int & count_rho);
  void                filterLogs(void);
  void                lookForSyntheticVsLog(float & rank_correlation);
  void                findILXLAtStartPosition(void);
  void                calculateDeviation(float  & devAngle,
                                         Simbox * timeSimbox);
  void                countFacies(Simbox *simbox, int * faciesCount);
  void                writeWell(int wellFormat);
  void                writeRMSWell(void);
  void                writeNorsarWell(void);
  void                moveWell(Simbox*timeSimbox, double deltaX, double deltaY, float kMove);

  static void         applyFilter(float *log_filtered, float *log_interpolated, int nt, double dt_milliseconds, float maxHz);
  static void         interpolateLog(float *log_interpolated, const float *log_raw, int nd);
  int                 isFaciesOk(){return faciesok_;};

private:
  void                readRMSWell(const std::string & wellFileName, const std::vector<std::string> & logNames,
                                  const std::vector<bool>  & inverseVelocity, bool porosityLogGiven, bool faciesLogGiven);
  void                readNorsarWell(const std::string & wellFileName, const std::vector<std::string> & logNames,
                                     const std::vector<bool>  & inverseVelocity, bool porosityLogGiven, bool faciesLogGiven);
  void                readLASWell(const std::string & wellFileName, const std::vector<std::string> & logNames,
                                  const std::vector<bool>  & inverseVelocity, bool porosityLogGiven, bool faciesLogGiven);
  void                processNRLibWell(const NRLib::Well              & well,
                                       const std::string              & wellFileName,
                                       const std::vector<std::string> & logNames,
                                       const std::vector<bool>        & inverseVelocity,
                                       bool                             porosityLogGiven,
                                       bool                             faciesLogGiven,
                                       bool                             norsarWell);
  void                mergeCells(const std::string & name, double * log_resampled, double * log,
                                 int ii, int istart, int iend, bool debug);
  void                mergeCells(const std::string & name, float * log_resampled, float * log,
                                int ii, int istart, int iend, bool debug);
  void                mergeCells(const std::string & name, int * log_resampled, int * log,
                                 int ii, int istart, int iend, bool debug);
  void                mergeCellsDiscrete(const std::string & name, int * log_resampled, int * log, int ii,
                                         int istart, int iend, bool printToScreen);
  bool                resampleTime(double * time_resampled, int nd, double & dt); //True if monotonously increasing well.
                                                                                  //Otherwise, no resampling done.
  void                resampleLog(float * log_resampled, const float * log_interpolated,
                                  const double * time, const double * time_resampled,
                                  int nd, double dt);

  ModelSettings           * modelSettings_;

  std::string               wellname_;
  std::string               wellfilename_;                // wellname given in RMS well file

  double                    xpos0_;                       // x-coordinate from well file header
  double                    ypos0_;                       // y-coordinate from well file header
  double                  * xpos_;                        // x-coord. in well
  double                  * ypos_;                        // y-coord in well
  double                  * zpos_;                        // time step
  double                  * md_;

  float                   * alpha_;
  float                   * beta_;
  float                   * rho_;
  int                     * facies_;
  float                   * porosity_;

  float                   * alpha_background_resolution_; // Vp  - filtered to background resolution
  float                   * beta_background_resolution_;  //
  float                   * rho_background_resolution_;   //

  float                   * alpha_seismic_resolution_;    // Vp  - filtered to seismic resolution
  float                   * beta_seismic_resolution_;     //
  float                   * rho_seismic_resolution_;      //

  BlockedLogs             * blockedLogsOrigThick_;
  BlockedLogs             * blockedLogsConstThick_;
  BlockedLogs             * blockedLogsExtendedBG_;

  int                       useForFaciesProbabilities_;   //Uses the indicator enum from Modelsettings.
  int                       useForFiltering_;             //Uses the indicator enum from Modelsettings.
  int                       useForWaveletEstimation_;     //Uses the indicator enum from Modelsettings.
  int                       useForBackgroundTrend_;       //Uses the indicator enum from Modelsettings.

  std::string               errTxt_;
  int                       error_;
  int                       timemissing_;
  int                       realVsLog_;                   //Uses the indicator enum from Modelsettings.
  bool                      isDeviated_;
  float                     meanVsVp_;                    // Average Vs/Vp for this well
  int                       nVsVp_;                       // Number of samples behind Vs/Vp estimate

  int                       nFacies_;
  std::string               faciesLogName_;
  std::vector<std::string>  faciesNames_;
  int                     * faciesNr_;
  int                       nd_;                          // number of obs. in well
  int                       faciesok_;                    // all faciesnumbers read are present in header

};

inline const float* WellData::getPorosity(int &nData) const{
  nData = nd_;
  return porosity_;
}

inline const float* WellData::getAlpha(int &nData) const
{
  nData = nd_;
  return alpha_;
}

inline const float* WellData::getBeta(int &nData) const
{
  nData = nd_;
  return beta_;
}

inline const float* WellData::getRho(int &nData) const
{
  nData = nd_;
  return rho_;
}
inline const int* WellData::getFacies(int &nData) const
{
  nData = nd_;
  return facies_;
}
inline const float* WellData::getAlphaBackgroundResolution(int &nData) const
{
  nData = nd_;
  return alpha_background_resolution_;
}

inline const float* WellData::getBetaBackgroundResolution(int &nData) const
{
  nData = nd_;
  return beta_background_resolution_;
}

inline const float* WellData::getRhoBackgroundResolution(int &nData) const
{
  nData = nd_;
  return rho_background_resolution_;
}

inline const float* WellData::getAlphaSeismicResolution(int &nData) const
{
  nData = nd_;
  return alpha_seismic_resolution_;
}

inline const float* WellData::getBetaSeismicResolution(int &nData) const
{
  nData = nd_;
  return beta_seismic_resolution_;
}

inline const float* WellData::getRhoSeismicResolution(int &nData) const
{
  nData = nd_;
  return rho_seismic_resolution_;
}


inline const double * WellData::getXpos(int &nData) const
{
  nData = nd_;
  return xpos_;
}
inline const double * WellData::getYpos(int &nData) const
{
  nData = nd_;
  return ypos_;
}
inline const double * WellData::getZpos(int &nData) const
{
  nData = nd_;
  return zpos_;
}
inline const double * WellData::getMD(int &nData) const
{
  if(md_ != NULL)
    nData = nd_;
  else
    nData = 0;
  return md_;
}
inline int WellData::getNd() const
{
  return nd_;
}
inline bool WellData::isFaciesLogDefined() const
{
  if(nFacies_==0)
    return 0;
  else
    return 1;
}
#endif
