//
// Created by mlakata on 8/24/20.
//

#ifndef PA_WS_OOMSCOREMONITOR_H
#define PA_WS_OOMSCOREMONITOR_H

#include <pacbio/logging/Logger.h>
#include <fstream>
#include <string>

/// A class for watching the OOM (Out of Memory) score. The Linux operating
/// system assigns a score to each process, and when the system gets close to running out of
/// memory, it kills the process with the highest score.
/// It is useful to know when the OOM score starts increasing, as this may demonstrate a memory leak or
/// design flaw.
class OomScoreMonitor
{
public:
    OomScoreMonitor() = default;

    /// \returns the OOM score for this process
    int32_t GetOomScore()
    {
        std::string oom_score_file = "/proc/self/oom_score";
        std::ifstream oom_score(oom_score_file);
        int32_t oomScore = 0;
        oom_score >> oomScore;
        return oomScore;
    }

    /// Looks for changes to the OOM score
    /// \param newScore - output pointer to latest OOM score. May be nullptr to ignore.
    /// \returns true if the score has changed
    bool PollOomScore(int32_t* newScore = nullptr)
    {
        int32_t score = GetOomScore();
        if (newScore) *newScore = score;
        if (score != lastOomScore_)
        {
            PBLOG_NOTICE << "OOM score went from " << lastOomScore_ << " to " << score;
            lastOomScore_ = score;
            return true;
        }
        return false;
    }
private:
    int32_t lastOomScore_ = -1;
};


#endif //PA_WS_OOMSCOREMONITOR_H
