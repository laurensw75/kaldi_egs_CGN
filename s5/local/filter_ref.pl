#!/usr/bin/perl
while(<STDIN>) {
    chop;
    s/'t/het/g;
    s/\[\S+\]//g;
    s/\.\.+//g;
    s/\*\S\s/ /g;
    s/ggg//g;
    s/xxx//g;
    s/\s+/ /g;
    print "$_\n";
}