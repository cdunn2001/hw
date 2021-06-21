#! env perl

# converts C++ structs to OpenAPI yaml schemas

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
"url"=>"string",   # fixme
"ControlledString_t"=>"string", #fixme
"ISO8601_Timestamp_t"=>"string" #fixme
);

#"std::vector<std::string>" => "[ string, string ...]",
#"std::vector<url>" => "[ url, url ...]",
#"std::vector<std::vector<double>>" => "[ [ double, double ...], [ double, double ...] ... ]",
#"std::vector<std::vector<int>>" => "[ [ int, int ...], [ int, int ...] ... ]",

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
            if ($_ =~ m|///<(.+)|)
            {
               $doc = $1;
               $doc =~ s/EXAMPLE\(.*\)//;     
            }

            my $f =  [ ];

            if (exists $enums{$class})
            {
                $f->[0] = $name;
                $f->[1] = "string";
                $f->[2] = "enum: [" .join(", ", @{$enums{$class}} )."]";
                $f->[3] = "example: \"" . $enums{$class}->[0]. "\"";
                $f->[4] = "description: $doc\n";
            }
            else
            {
                $class = $subs{$class} || $class;
                $class = transmogrify($class);
                $f->[0] = $name;
                $f->[1] = $class;
                $f->[3] = "example: $example";
                $f->[4] = "description: $doc\n";
            }
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

