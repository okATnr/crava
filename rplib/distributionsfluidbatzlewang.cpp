#include "rplib/distributionsfluidbatzlewang.h"

#include "rplib/distributionwithtrend.h"
#include "rplib/fluidbatzlewang.h"
#include "rplib/demmodelling.h"

#include "nrlib/random/distribution.hpp"

DistributionsFluidBatzleWang::DistributionsFluidBatzleWang(DistributionWithTrend        * distr_temperature,
                                                           DistributionWithTrend        * distr_pore_pressure,
                                                           DistributionWithTrend        * distr_salinity,
                                                           std::vector<double>          & alpha)
: DistributionsFluid()
{
  if (distr_temperature->GetIsShared() == true)
    distr_temperature_ = distr_temperature;
  else
    distr_temperature_ = distr_temperature->Clone();

  if (distr_pore_pressure->GetIsShared() == true)
    distr_pore_pressure_ = distr_pore_pressure;
  else
    distr_pore_pressure_ = distr_pore_pressure->Clone();

  if (distr_salinity->GetIsShared() == true)
    distr_salinity_ = distr_salinity;
  else
    distr_salinity_ = distr_salinity->Clone();

  alpha_               = alpha;
}

DistributionsFluidBatzleWang::DistributionsFluidBatzleWang(const DistributionsFluidBatzleWang & dist)
: DistributionsFluid(dist)
{
  if(dist.distr_salinity_->GetIsShared() == false)
    distr_salinity_ = dist.distr_salinity_->Clone();
  else
    distr_salinity_ = dist.distr_salinity_;

  if(dist.distr_temperature_->GetIsShared() == false)
    distr_temperature_ = dist.distr_temperature_->Clone();
  else
    distr_temperature_ = dist.distr_temperature_;

  if(dist.distr_pore_pressure_->GetIsShared() == false)
    distr_pore_pressure_ = dist.distr_pore_pressure_->Clone();
  else
    distr_pore_pressure_ = dist.distr_pore_pressure_;

  alpha_               = dist.alpha_;
}

DistributionsFluidBatzleWang::~DistributionsFluidBatzleWang()
{
  if(distr_salinity_->GetIsShared() == false)
    delete distr_salinity_;

  if(distr_temperature_->GetIsShared() == false)
    delete distr_temperature_;

  if(distr_pore_pressure_->GetIsShared() == false)
    delete distr_pore_pressure_;
}

DistributionsFluid *
DistributionsFluidBatzleWang::Clone() const
{
  return new DistributionsFluidBatzleWang(*this);
}

Fluid *
DistributionsFluidBatzleWang::GenerateSample(const std::vector<double> & trend_params)
{
  std::vector<double> u(3);
  for(int i=0; i<3; i++)
    u[i] = NRLib::Random::Unif01();

  Fluid * fluid = GetSample(u, trend_params);

  return  fluid;
}

Fluid *
DistributionsFluidBatzleWang::GetSample(const std::vector<double> & u,
                                        const std::vector<double> & trend_params)
{
  double  salinity      = distr_salinity_     ->GetQuantileValue(u[0], trend_params[0], trend_params[1]);
  double  temperature   = distr_temperature_  ->GetQuantileValue(u[1], trend_params[0], trend_params[1]);
  double  pore_pressure = distr_pore_pressure_->GetQuantileValue(u[2], trend_params[0], trend_params[1]);

  Fluid * fluid         = new FluidBatzleWang(salinity, temperature, pore_pressure, u);

  return  fluid;
}

bool
DistributionsFluidBatzleWang::HasDistribution() const
{
  bool has_distribution = false;

  if(distr_salinity_->GetIsDistribution() == true || distr_temperature_->GetIsDistribution() || distr_pore_pressure_->GetIsDistribution() == true)
    has_distribution = true;

  return(has_distribution);
}

std::vector<bool>
DistributionsFluidBatzleWang::HasTrend() const
{
  std::vector<bool> has_trend(2, false);

  std::vector<bool> salinity_trend    = distr_salinity_     ->GetUseTrendCube();
  std::vector<bool> temperature_trend = distr_temperature_  ->GetUseTrendCube();
  std::vector<bool> pressure_trend    = distr_pore_pressure_->GetUseTrendCube();

  for(int i=0; i<2; i++) {
    if(salinity_trend[i] == true || temperature_trend[i] == true || pressure_trend[i] == true)
      has_trend[i] = true;
  }

  return has_trend;
}

Fluid *
DistributionsFluidBatzleWang::UpdateSample(double                      corr_param,
                                           bool                        param_is_time,
                                           const std::vector<double> & trend,
                                           const Fluid               * sample)
{
  std::vector<double> u = sample->GetU();
  DEMTools::UpdateU(u, corr_param, param_is_time, alpha_);
  Fluid * updated_sample = GetSample(u, trend);

  return updated_sample;
}
