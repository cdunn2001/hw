# TODO. ported this

class Verification:
    bam2bamExe = "bam2bam"
    bamPaVerifyExe = "bampaverify"

    def __init__(self):
        self.inputPrefix = ''
        self.outputPrefix = ''

    def Run(self, limsPrefix, darkFrameCalFile):
        global args
        self.inputPrefix = limsPrefix
        self.outputPrefix = os.path.join(args.outputprefix, os.path.basename(limsPrefix))
        logging.info("Verification.Run: inputPrefix = %s, outputPrefix = %s", self.inputPrefix, self.outputPrefix)

        if args.zmwmatchthreshold > 0:
            self.RunBam2Bam()
            self.RunBamPaVerify()
            errors = self._countErrors()
            if errors <= args.zmwmatchthreshold:
                pass
            else:
                raise Exception("Verification of BAM files failed, errors =", errors,
                                " exceeded threshold of ", args.zmwmatchthreshold)
        else:
            logging.info("Skipping BAM verification")

        if args.verifymovie:
            self.RunDeconv(darkFrameCalFile)
            self.RunMovieDiff()

    def RunDeconv(self, darkFrameCalFile):
        global args
        command = [
            args.deconv,
            '--src', args.datafile,
            '--drk', darkFrameCalFile,
            '--filter', '/pbi/dept/primary/sim/sequel/xtalk.txt',
            '--frames', '512',
            '--dst', self.outputPrefix + '.expected.mov.h5'
        ]
        logging.info("Verification.RunDeconv: %s", command)
        check_call(command)

    def RunPbValidate(self, subreadsetXml):
        command = [
            'pbvalidate',
            '--quick',
            subreadsetXml
        ]
        logging.info("Verification.RunPbValidate: %s", command)
        check_call(command)

    def RunMovieDiff(self):
        global args
        command = [args.sequel_movie_diff,
                   '--diff',
                   '--frames', '512',
                   '--dataonly',
                   self.outputPrefix + '.expected.mov.h5',
                   self.inputPrefix + '.mov.h5'
                   ]
        try:
            logging.info('Verification.RunMovieDiff command: %s', ' '.join(command))
            diff_output = check_output(command).decode(sys.stdout.encoding)
        except subprocess.CalledProcessError as e:
            raise Exception("Verification of movie file failed, errors=\n", e.output)

    def RunBam2Bam(self):
        command = [Verification.bam2bamExe,
                   '-s',
                   self.inputPrefix + '.subreadset.xml',
                   '--zmw',
                   '-o',
                   self.outputPrefix
                   ]
        logging.info("Verification.RunBam2Bam: %s", command)
        check_call(command)

    def RunBamPaVerify(self):
        command = [Verification.bamPaVerifyExe,
                   '--outputFile',
                   self.outputPrefix + '.csv',
                   self.outputPrefix + '.zmws.bam'
                   ]
        logging.info("Verification.RunBamPaVerify: %s", command)
        check_call(command)

    def _countErrors(self):
        command = ['wc', '-l', self.outputPrefix + '.csv']
        logging.info('Verification._countErrors: %s' , ' '.join(command))
        s = check_output(command).decode(sys.stdout.encoding)
        count = int(s.split(' ')[0])
        count = count - 1
        logging.info("Verification._countErrors: mismatches = %d", count)
        return count