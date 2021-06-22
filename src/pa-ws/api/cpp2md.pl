#! env perl

use strict;

my $instruct = 0;

my %enums;
my $currentStruct = "";
my $indent = 0;
my @fields;

my %subs = ("std::string" => "string",
"std::vector<std::string>" => "[ string, string ...]",
"std::vector<url>" => "[ url, url ...]",
"std::vector<std::vector<double>>" => "[ [ double, double ...], [ double, double ...] ... ]",
"std::vector<std::vector<int>>" => "[ [ int, int ...], [ int, int ...] ... ]",
);

sub transmogrify
{
    my ($s) = @_;

    if ($s =~ /std::vector<(.+)>/)
    {   
        my $inner = $1;
        if ($inner =~ /std::vector/) { $inner = transmogrify($inner); }
        return " [ $inner, $inner ... ]";
    }
    else
    {
        return $s;
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
            print join(",\n", @fields)."\n";
            $indent--;
            print "}  \n\n";
            $currentStruct  = "";
            $instruct = 0;
        }
        elsif (/PB_CONFIG\(\s*(\S+)\);/)
        {

        }
        elsif (/PB_CONFIG_PARAM\(\s*(\S+),\s*(\S+),\s*(\S+)\s*\);/)
        {
            my ($class, $name, $def) = ($1,$2,$3);

            my $f = "  " x $indent;

            if (exists $enums{$class})
            {
                $f .= "\"$name\": " . join(" | ",map "\"$_\"", @{$enums{$class}});
            }
            else
            {
                $class = $subs{$class} || $class;
                $class = transmogrify($class);
                $f .= "\"$name\": $class";
            }
            push @fields, $f;
        }
        elsif (/PB_CONFIG_OBJECT\(\s*(\S+),\s*(\S+)\s*\);/)
        {
            my ($class, $name) = ($1,$2);

            my $f = "  " x $indent;

            $f .= "\"$name\": $class";
            push @fields, $f;

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
            print "$currentStruct := {\n";
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
            print "$class := $x $example\n\n";
        }
        else
        {
          #  print("ignoring fluff: $_");
        }
    }
}

use Data::Dumper;

# print Dumper(\%enums);

