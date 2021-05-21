#ifndef SEQUEL_BARCODESTRATEGY_H
#define SEQUEL_BARCODESTRATEGY_H

namespace PacBio {
namespace Primary {
namespace Postprimary {

// Enum encodes the two available barcoding strategies
enum class BarcodeStrategy
{
    SYMMETRIC,
    ASYMMETRIC,
    TAILED
};

}}} // ::PacBio::Primary::Postprimary

#endif //SEQUEL_BARCODESTRATEGY_H
