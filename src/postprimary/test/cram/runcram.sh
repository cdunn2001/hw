. /etc/profile.d/modules.sh
module use /mnt/software/modulefiles
module use /pbi/dept/primary/modulefiles
module purge
module load parallel cram/0.6
module load smrttools/8.0.0
module load samtools
module load xpath
module load jq
module load primary-toolkit/1.0.7


if [[ "$DEBUG" == 1 ]]; then
 echo "Loading PPA Debug build"
 module load ppa-mongo/workspace/debug
else
 module load ppa-mongo/workspace
fi

echo "Using baz2bam: $(which baz2bam)."
echo "Using bam2bam: $(which bam2bam)."
echo "Using simbazwriter: $(which simbazwriter)."
echo "Using bazviewer: $(which bazviewer)."

unset TERM
fail=0
tests="recalladapters.t bam2bam_nobc_snr_recovery.t bam2bam_nobc_rl_recovery.t baz2bam_stsh5.t bam2bamCtrl_internal.t bam2bamCtrl.t bam2bamCtrl_validate_internal.t bam2bamCtrl_validate.t bam2bam_internal.t bam2bam.t bam2bam_spider.t bam2bam_validate_internal.t bam2bam_validate.t baz2bam.t baz2bam_validate_end2end.t baz2bam_metrics.t baz2bam_startFrame.t baz2bam_stsh5vsstsxml.t baz2bam_snr_recovery.t ppa-reducestats.t baz2bam_numseq.t baz2bam_pulse_exclusion.t adapter_correction.t wallStart_wallEnd.t baz2bam_metadatacfg.t barcoded_adapters.t"

# check that all the files exist
if ! ls $tests 1>/dev/null
then
 echo Some tests do not exist, cram tests not run
 exit 2 
fi

# ls $tests simply puts the filenames on separate lines, so that parallel can consume them
ls $tests | parallel -j 8 --no-notice cram -v || fail=1

exit $fail
