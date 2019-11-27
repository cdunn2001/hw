%load('\\usmp-data2\DATA\NanoFabricationDATA\CAD\00000011-MultiMillionProject(Sequel2)\20171018-IMEC-SpiderAlpha-MaskPrepRev05-Spider_1p0_NTO\Spider_Full_Array.mat')
load('\\usmp-data2\DATA\NanoFabricationDATA\CAD\00000011-MultiMillionProject(Sequel2)\20180611-PBI-PAtchingMatlabDocumentionForMathieuScrewUp-NotALayoutChange-DocumentationONLY\Spider_Full_Array.mat')

xes=Array(:,:,1);
yes=Array(:,:,2);
zes=Array(:,:,3);
nonzmws = find(zes ~= 1);
fileID = fopen('stuff3.c','w');
fprintf(fileID,'extern const size_t numNonZmws = %d;\n',numel(nonzmws));
fprintf(fileID,'const int16_t xes[] = {\n');

% Print 10 elements per line.
lineFormat = [repmat('%4d,', 1, 10) '\n'];

fprintf(fileID, lineFormat, xes(nonzmws));
fprintf(fileID,'};\n\n');

fprintf(fileID,'const int16_t yes[] = {\n');
fprintf(fileID, lineFormat, yes(nonzmws));
fprintf(fileID,'};\n\n');

fprintf(fileID,'const uint8_t zes[] = {\n');
fprintf(fileID, lineFormat, zes(nonzmws));
fprintf(fileID,'};\n');

fclose(fileID);
