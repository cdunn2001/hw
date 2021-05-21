


BAM file size

see byte_per_base_Bases_Without_QVs.png
445 files over 100MB

    const double BaseReadoutBytesPerBase_mean = 1.952;
    const double BaseReadoutBytesPerBase_stddev= 0.028;
    const double BaseReadoutBytesPerBase_limit = BaseReadoutBytesPerBase_mean + 5*BaseReadoutBytesPerBase_stddev;


see byte_per_base_Pulses.png
9103 files between 1GB and 5GB surveyed

These values represent *average* bytes per pulse, which also include overhead, metrics, headers, padding.
    const double PulseReadoutBytesPerBase_mean   = 9.877;
    const double PulseReadoutBytesPerBase_stddev = 0.341;
    const double PulseReadoutBytesPerBase_limit  =  PulseReadoutBytesPerBase_mean + 5*PulseReadoutBytesPerBase_stddev;


see bytes_per_pulse.png
This is 10000 ZMWS, and shows that the marginal bytes per pulse is higher in some cases, 13.7 bytes/pulse
