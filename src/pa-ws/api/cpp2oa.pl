#! env perl

# Copyright (c) 2021, Pacific Biosciences of California, Inc.
#
# All rights reserved.
#
# THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
# AND PROPRIETARY INFORMATION.
#
# Disclosure, redistribution and use of this software is subject to the
# terms and conditions of the applicable written agreement(s) between you
# and Pacific Biosciences, where "you" refers to you or your company or
# organization, as applicable.  Any other disclosure, redistribution or
# use is prohibited.
#
# THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# File Description:
#  \brief converts C++ structs to OpenAPI yaml schemas
# 
# The script should be given a list of all headers used for the API as command line
# parameters.
# The header file format must strict conform to the requirements of this script
# 
# 1. The structs must be derived from PBConfig
#
# 2. Each parameter and Oxygen comments must be on a single line.
#
# 3. Each parameter should have ///< documentation at the end of the line
#
# 4. Inside the ///< documentation, there should an EXAMPLE() macro that gives a raw example
#    of the parameter.
#
# Example:
#
#    PB_CONFIG_PARAM(double, progress, 0.0); ///< progress of job completion. Range is [0.0, 1.0] EXAMPLE(0.74)


use strict;

my $instruct = 0;

my %enums;
my $currentStruct = "";
my $indent = 0;
my @fields;
my %objects;

my %subs = ("std::string" => "string",
"double"=>"number",
"int"=>"integer",
"uint32_t"=>"integer",
"uint64_t"=>"integer",
"bool"=>"boolean",
"url"=>"string",   # FIXME - This richer type information should be shown in the OpenAPI description, rather than simply give "string" as the type.
"ControlledString_t"=>"string", # FIXME
"ISO8601_Timestamp_t"=>"string" # FIXME
);

sub transmogrify
{
    my ($s) = @_;

    if ($s =~ /std::vector<(.+)>/)
    {   
        my $inner = $1;
        my @inners;
        if ($inner =~ /std::vector/) { 
            $inner = transmogrify($inner); 
            if (ref($inner) eq 'ARRAY') { push @inners, @$inner; }
            else { push @inners, $inner; }
        }
        else
        {
            push @inners, $inner;
        }
        
        return [ "array", @inners ];
    }
    else
    {
        return $s;
    }
}
print GetIndent($indent) . "components:\n";
$indent++;
print GetIndent($indent) . "schemas:\n";
$indent++;

sub GetIndent
{
    my ($indent0) = @_;
    return "  " x $indent0;
}
sub ShowStruct
{
    my ($indent1, $ffields) = @_;

    my $ii = GetIndent($indent1);

    print $ii . "properties:\n";

    $indent1++;

    for my $f (@$ffields)
    {
        $ii = GetIndent($indent1);
        print $ii . "$f->[0]:\n";

        $indent1++;
        $ii = GetIndent($indent1);

        my $type = $f->[1];
        if (ref($type) eq 'ARRAY')
        {
            my $indent2= $indent1;
            while (scalar(@$type) > 0)
            {
                $indent2++;
                $ii = GetIndent($indent2);
                my $k = shift @$type;
                if ($k eq "array")
                {
                    print $ii . "type: array\n";
                    print $ii . "items:\n";
                }
                elsif ($k =~ /Object/)
                {
                    print $ii . "\$ref: '#/components/schemas/$k'\n";
                }
                else
                {
                    $k = $subs{$k} || $k;
                    print $ii . "type: $k\n"
                }
            }
        }
        elsif ($type =~ /Object/) {
            $indent1++;
            $ii = GetIndent($indent1);
            print $ii . "type: object\n";
            if (!$objects{$type})
            {
                die "object $type not defined";
            }
            ShowStruct($indent1, $objects{$type});
            $indent1--;
        }
        else
        {
            print $ii . "type: $type\n";
        }
        if ($f->[2])
        {
            print $ii . "$f->[2]\n";
        }
        if ($f->[3])
        {
            print $ii . "$f->[3]\n";
        }
        if ($f->[4])
        {
            print $ii . "$f->[4]\n";
        }
        $indent1--;
    }
    $indent1--;
    $ii = GetIndent($indent1);
    print $ii . "required:\n";
    for my $f (@fields)
    {
        print $ii . " - $f->[0]\n";
    }
}

while(<>)
{
    if (/SMART_ENUM\((.*)\);/)
    {
        my @x =split /\s*,\s*/, $1;
        my $enum = shift @x;
        $enums{$enum} = [ @x ];
    }
    elsif ($instruct)
    {
        if (/^\s*}/)
        {
            ShowStruct($indent, \@fields);

            $indent--;
            
            $currentStruct  = "";
            $instruct = 0;
        }
        elsif (/PB_CONFIG\(\s*(\S+)\);/)
        {

        }
        elsif (/PB_CONFIG_PARAM\(\s*(\S+),\s*(\S+),\s*(\S+)\s*\);/)
        {
            my ($class, $name, $def) = ($1,$2,$3);

            my $example;
            if (/EXAMPLE\((.+)\)/)
            {
                $example = $1;
            }
            my $doc = "";
            if ($_ =~ m|///<\s*(.+)|)
            {
               $doc = $1;
               $doc =~ s/\s*EXAMPLE\(.+\)//;     
            }

            my $f =  [ ];

            $f->[0] = $name;
            if (exists $enums{$class})
            {
                $f->[1] = "string";
                $f->[2] = "enum: [" .join(", ", @{$enums{$class}} )."]";
                $example ||= "\"" . $enums{$class}->[0]. "\"";
            }
            else
            {
                $class = $subs{$class} || $class;
                $class = transmogrify($class);
                $f->[1] = $class;
            }
            $f->[3] = "example: $example";
            $doc =~ s/\'/\\'/g;
            $f->[4] = "description: '$doc'\n";
            push @fields, $f;
            push @{$objects{$currentStruct}}, $f;

        }
        elsif (/PB_CONFIG_OBJECT\(\s*(\S+),\s*(\S+)\s*\);/)
        {
            my ($class, $name) = ($1,$2);

            my $f = [ ];

            $f->[0] = $name;
            $f->[1] = $class;
            push @fields, $f;
            push @{$objects{$currentStruct}}, $f;
        }
        elsif (/PB_CONFIG_OBJECT_WITH_DEF\(\s*(\S+),\s*(\S+),\s*(\S+),\s*\);/)
        {
            my ($class, $name, $def) = ($1,$2);

            my $f = [ ];

            $f->[0] = $name;
            $f->[1] = $class;
            push @fields, $f;
            push @{$objects{$currentStruct}}, $f;
        }
        elsif (/^\s*$/)
        {
            # blank line
        }
        elsif (m|^\s*//|)
        {
            # comment
        }
        else
        {
            print("BAD LINE! $_");
        }
    }
    else
    {
        if (/^struct\s+(\S+)\s.*PBConfig/)
        {
            $currentStruct = $1;
            print GetIndent($indent);
            print "$currentStruct:  #top level\n";
        }
        elsif (/^\s*{/)
        {
            $instruct = 1;
            $indent++;
            @fields = ();
        }
        elsif (/^typedef\s+(\S+)\s+(\S+);/)
        {
            my ($x,$class) = ($1,$2);
            $x = $subs{$x} || $x;
            my $example = "";
            if (/EXAMPLE\((.+)\)/)
            {
                $example = "   # e.g. $1";
            }
            if (0)
            {
                print "    $class:\n";
                print "      properties:\n";
                print "        $x:\n";
                print "          type: string\n\n";  # fixme
                print "          example: $example\n\n";
            }
        }
        else
        {
          #  print("ignoring fluff: $_");
        }
    }
}

use Data::Dumper;

# print Dumper(\%enums);

