/***************************************************************************
*      Copyright (C) 2008 by Norwegian Computing Center and Statoil        *
***************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define _USE_MATH_DEFINES
#include <cmath>

//#include "src/commondata.h"
#include "src/seismicstorage.h"
#include "src/definitions.h"
//#include "src/simbox.h"

SeismicStorage::SeismicStorage()
{
}

SeismicStorage::SeismicStorage(std::string   filename,
                               int           seismic_type,
                               float         angle,
                               NRLib::SegY * segy)
 : file_name_(filename),
   angle_(angle),
   seismic_type_(seismic_type),
   segy_(segy)
{
}

SeismicStorage::SeismicStorage(std::string     filename,
                               int             seismic_type,
                               float           angle,
                               StormContGrid * storm_grid)
 : file_name_(filename),
   angle_(angle),
   seismic_type_(seismic_type),
   storm_grid_(storm_grid)
{
}

SeismicStorage::~SeismicStorage()
{
}

std::vector<float>
SeismicStorage::GetTraceData(int index) const
{
  //Return a vector(float) of trace data for trace index.

  std::vector<float> trace_data;

  if(seismic_type_ == SEGY) {
    NRLib::SegYTrace * segy_tmp = segy_->getTrace(index);

    if(segy_tmp != NULL) {
      size_t start = segy_tmp->GetStart();
      size_t end = segy_tmp->GetEnd();
      for(size_t i = start; i < end; i++)
        trace_data.push_back(segy_tmp->GetValue(i));
    }
  }
  else {

    double x_tmp = 0.0;
    double y_tmp = 0.0;
    double z_tmp = 0.0;

    size_t i;
    size_t j;
    size_t k;

    storm_grid_->GetIJK(index, i, j, k);

    for(size_t kk = 0; kk < storm_grid_->GetNK(); kk++) {
      storm_grid_->FindCenterOfCell(i, j, kk, x_tmp, y_tmp, z_tmp);
      trace_data.push_back(storm_grid_->GetValueClosestInZ(x_tmp, y_tmp, z_tmp));
    }
  }

  return trace_data;
}


void
SeismicStorage::GetSparseTraceData(std::vector<std::vector<float> > & trace_data,
                                   std::vector<float>               & trace_length,
                                   int                                n)
{
  //Return a vector(float) of trace data with n elements.
  //SEGY: n traces. STORM/SGRI: n grid-elements.

  if(seismic_type_ == SEGY) {
    int n_traces = segy_->GetNTraces();

    if(n > n_traces)
      n = n_traces;

    trace_length.resize(n, 1);
    trace_data.resize(n);

    for(int i = 0; i < n; i++) {

      int trace_index = i*(static_cast<int>(n_traces / n));

      NRLib::SegYTrace * segy_tmp = segy_->getTrace(trace_index);

      if(segy_tmp != NULL) {
        size_t start = segy_tmp->GetStart();
        size_t end = segy_tmp->GetEnd();
        for(size_t j = start; j < end; j++)
          trace_data[i].push_back(segy_tmp->GetValue(j));

      }
    }
  }
  else {

    double x_tmp = 0.0;
    double y_tmp = 0.0;
    double z_tmp = 0.0;

    int index_i = 0;
    int index_j = 0;
    int trace_index = 0;

    int n_elements = static_cast<int>(std::sqrt(static_cast<double>(n)));

    for(int i = 0; i < n_elements; i++) {

      index_i = i*static_cast<size_t>(storm_grid_->GetNI()/n_elements);
      if(index_i >= storm_grid_->GetNI())
        index_i = storm_grid_->GetNI() -1;

      for(int j = 0; j < n_elements; j++) {

        index_j = j*static_cast<size_t>(storm_grid_->GetNJ()/n_elements);
        if(index_j >= storm_grid_->GetNJ())
          index_j = storm_grid_->GetNJ()-1;

        for(size_t k = 0; k < storm_grid_->GetNK(); k++) {
          storm_grid_->FindCenterOfCell(index_i, index_j, k, x_tmp, y_tmp, z_tmp);
          trace_data[trace_index].push_back(storm_grid_->GetValueClosestInZ(x_tmp, y_tmp, z_tmp));
        }

        //Store length
        double top = storm_grid_->GetTopSurface().GetZ(x_tmp, y_tmp);
        double bot = storm_grid_->GetBotSurface().GetZ(x_tmp, y_tmp);
        trace_length.push_back(std::abs(bot-top));

        trace_index++;
      }
    }
  }
}

std::vector<float>
SeismicStorage::GetRealTrace(const Simbox * estimation_simbox,
                             int i,
                             int j) const
{
  std::vector<float> value;
  double x_tmp; double y_tmp; double z_tmp;

  if(seismic_type_ == SEGY) {
    for(size_t k = 0; k < segy_->GetNz(); k++) {
      estimation_simbox->getCoord(i, j, k, x_tmp, y_tmp, z_tmp);
      value.push_back(segy_->GetValue(x_tmp, y_tmp, z_tmp)); ///H Is this correct? Want to get det same value as FFTGrid::getRealTrace2(int i, int j)
    }
  }
  else {
    for(size_t k = 0; k < storm_grid_->GetNK(); k++) {
      storm_grid_->FindCenterOfCell(i, j, k, x_tmp, y_tmp, z_tmp);
      value.push_back(storm_grid_->GetValueClosestInZ(x_tmp, y_tmp, z_tmp));
    }
  }

  return value;
}

float
SeismicStorage::GetRealTraceValue(const Simbox * estimation_simbox,
                                  int i,
                                  int j,
                                  int k) const
{
  float value;
  double x_tmp; double y_tmp; double z_tmp;

  if(seismic_type_ == SEGY) {
    estimation_simbox->getCoord(i, j, k, x_tmp, y_tmp, z_tmp);
    value = segy_->GetValue(x_tmp, y_tmp, z_tmp);
  }
  else {
    storm_grid_->FindCenterOfCell(i, j, k, x_tmp, y_tmp, z_tmp);
    value = storm_grid_->GetValueClosestInZ(x_tmp, y_tmp, z_tmp);
  }

  return value;
}
