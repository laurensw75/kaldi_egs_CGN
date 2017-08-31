#!/bin/bash

# This script trains an LM on the CGN LM-training data.

dict_suffix=

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

dir=data/local/local_lm
srcdir=data/local/dict${dict_suffix}

mkdir -p $dir
. ./path.sh || exit 1; # for KALDI_ROOT
export PATH=$KALDI_ROOT/tools/kaldi_lm:$PATH
( # First make sure the kaldi_lm toolkit is installed.
	cd $KALDI_ROOT/tools || exit 1;
	if [ -d kaldi_lm ]; then
		echo Not installing the kaldi_lm toolkit since it is already there.
	else
		echo Downloading and installing the kaldi_lm tools
		if [ ! -f kaldi_lm.tar.gz ]; then
			wget http://www.danielpovey.com/files/kaldi/kaldi_lm.tar.gz || exit 1;
		fi
		tar -xvzf kaldi_lm.tar.gz || exit 1;
		cd kaldi_lm
		make || exit 1;
		echo Done making the kaldi_lm tools
	fi
) || exit 1;


if [ ! -f $srcdir/cleaned.gz -o ! -f $srcdir/lexicon.txt ]; then
	echo "Expecting files $srcdir/cleaned.gz and $srcdir/lexicon.txt to exist";
	exit 1;
fi

# Get a wordlist-- keep everything but silence, which should not appear in
# the LM.
awk '{print $1}' $srcdir/lexicon.txt | grep -v -w '!SIL' > $dir/wordlist.txt

# Get training data with OOV words (w.r.t. our current vocab) replaced with <UNK>.
echo "Getting training data with OOV words replaced with <unk> (train_nounk.gz)" 
gunzip -c $srcdir/cleaned.gz | awk -v w=$dir/wordlist.txt \
	'BEGIN{while((getline<w)>0) v[$1]=1;}
	{for (i=1;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}'|sed 's/ $//g' \
	| gzip -c > $dir/train_nounk.gz

# Get unigram counts (without bos/eos, but this doens't matter here, it's
# only to get the word-map, which treats them specially & doesn't need their
# counts).
# Add a 1-count for each word in word-list by including that in the data,
# so all words appear.
gunzip -c $dir/train_nounk.gz | cat - $dir/wordlist.txt | \
	awk '{ for(x=1;x<=NF;x++) count[$x]++; } END{for(w in count){print count[w], w;}}' | \
	sort -nr > $dir/unigram.counts

# Get "mapped" words-- a character encoding of the words that makes the common words very short.
cat $dir/unigram.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>" "<unk>" > $dir/word_map

gunzip -c $dir/train_nounk.gz | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
	{ for(n=1;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz

# To save disk space, remove the un-mapped training data.  We could
# easily generate it again if needed.
rm $dir/train_nounk.gz 

train_lm.sh --arpa --lmtype 3gram-mincount $dir
# Perplexity over 102584.000000 words (excluding 2158.000000 OOVs) is 149.685778
# 1.6 million N-grams.

prune_lm.sh --arpa 1.5 $dir/3gram-mincount/
# Perplexity over 102584.000000 words (excluding 2158.000000 OOVs) is 158.659433
# 0.48 million N-grams.

train_lm.sh --arpa --lmtype 4gram-mincount $dir
# Perplexity over 102584.000000 words (excluding 2158.000000 OOVs) is 146.859389
# 1.9 million N-grams.

prune_lm.sh --arpa 1.5 $dir/4gram-mincount
# Perplexity over 102584.000000 words (excluding 2158.000000 OOVs) is 156.526187
# 0.53 million N-grams

exit 0
