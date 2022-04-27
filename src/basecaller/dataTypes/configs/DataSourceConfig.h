// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef basecaller_dataTypes_configs_DataSourceConfig_H_
#define basecaller_dataTypes_configs_DataSourceConfig_H_

#include <pacbio/datasource/MovieInfo.h>
#include <pacbio/datasource/PacketLayout.h>

namespace PacBio::Mongo::Data
{
struct DataSourceConfig : public Configuration::PBConfig<DataSourceConfig>
{
    PB_CONFIG(DataSourceConfig);

    PB_CONFIG_PARAM(std::string, darkCalFileName, "");
    PB_CONFIG_PARAM(std::vector<std::vector<double>>, imagePsfKernel, 0);
    PB_CONFIG_PARAM(std::vector<std::vector<double>>, crosstalkFilterKernel, 0);
    // PB_CONFIG_PARAM(std::vector<int>, decimationMask, 0); TODO
};

} // namespace

namespace PacBio::Configuration {

template <>
inline void ValidateConfig<Mongo::Data::DataSourceConfig>(
        const Mongo::Data::DataSourceConfig& dataSource,
        ValidationResults* results)
{
    if (dataSource.crosstalkFilterKernel.size() > 0 && 
        dataSource.imagePsfKernel.size() > 0)
    {
        results->AddError("dataSource.crosstalkFilterKernel and dataSource.imagePsfKernel cannot be set at the same time. Pick one or the other.");
    }        


    auto validateKernel = [&results](const decltype(dataSource.crosstalkFilterKernel)& k, const std::string& name)
    {
        if (k.size() > 0)
        {
            if ((k.size() % 2) == 0)
            {
                results->AddError(name + " must have dimensions that are ODD. Outer dimension was " + std::to_string(k.size()));
            }
            else 
            {
                // check all inner vectors
                size_t expectedSize = k[0].size();
                for(const auto& kk : k)
                {
                    if(kk.size() ==0)
                    {
                        results->AddError(name + " must be a valid 2D vector of vectors. Inner vector was size 0");
                    } 
                    else if ((kk.size() % 2) == 0)
                    {
                        results->AddError(name + " must have dimensions that are ODD. Inner dimension was " + std::to_string(kk.size()));
                    } else if (expectedSize != kk.size())
                    {
                        results->AddError(name + " must have equal sized inner vectors. Mismatch in row sizes:" 
                            + std::to_string(expectedSize) + " vs " + std::to_string(kk.size()));
                    }
                }
            }
        }
    };

    validateKernel(dataSource.crosstalkFilterKernel,"dataSource.crosstalkFilterKernel");
    validateKernel(dataSource.imagePsfKernel,"dataSource.imagePsfKernel");
}

} // namespace

#endif // applications_dataTypes_configs_DataSourceConfig_H_
