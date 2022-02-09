input=$1
output=$2

if [[ $input == "" ]]
then
    echo Need input file name
    echo Usage: $0 input.md output.html
    exit 1
fi
if [[ $input == "" ]]
then
    echo need output file name
    echo Usage: $0 input.md output.html
    exit 1
fi

input=$(realpath $input)
output=$(realpath $output)

css=$(realpath styling.css)
css=styling.css
# css=$(realpath pandoc_style.ihtml)

workdir=$(mktemp -d)
pushd $workdir
cat > .puppeteer.json <<EOF
{ "args":["--no-sandbox"] }
EOF

set -x
# --toc generate table of contents
simg=/pbi/dept/deployment/simg/c7.6-mermaidcli.simg
# simg=/data/pb/c7.6-mermaidcli.simg
singularity run $simg pandoc -t html5 -F mermaid-filter -s --css $css -i $input -o $output
# singularity run /data/pb/c7.6-mermaidcli.simg pandoc -t html5 -F mermaid-filter -s --toc -H $css -i $input -o $output
popd
rm -rf $workdir
