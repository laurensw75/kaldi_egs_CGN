#!/bin/bash

# Preparation for CGN data by LvdW

if [ $# -le 2 ]; then
   echo "Arguments should be <CGN root> <language> <comps>, see ../run.sh for example."
   exit 1;
fi

cgn=$1
lang=$2
comps=$3

base=`pwd`
dir=`pwd`/data/local/data
lmdir=`pwd`/data/local/cgn_lm
dictdir=`pwd`/data/local/dict_nosp
mkdir -p $dir $lmdir
local=`pwd`/local
utils=`pwd`/utils

. ./path.sh 	# Needed for KALDI_ROOT

if [ -z $SRILM ] ; then
  export SRILM=$KALDI_ROOT/tools/srilm
fi
export PATH=${PATH}:$SRILM/bin/i686-m64
if ! command -v ngram-count >/dev/null 2>&1 ; then
  echo "$0: Error: SRILM is not available or compiled" >&2
  echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
  echo "$0: Error: and run extras/install_srilm.sh" >&2
  exit 1
fi

cd $dir

# create train & dev set
## Create .flist files (containing a list of all .wav files in the corpus)
rm -f temp.flist
IFS=';'
for l in $lang; do
	for i in $comps; do
		find ${cgn}/data/audio/wav/comp-${i}/${l} -name '*.wav' >>temp.flist
	done
done
IFS=' '
# now split into train and dev
grep -vF -f $local/nbest-dev-2008.txt temp.flist | grep 'comp-c\|comp-d' | sort >train_t.flist
grep -vF -f $local/nbest-dev-2008.txt temp.flist | grep -v 'comp-c\|comp-d' | sort >train_s.flist
grep -F -f $local/nbest-dev-2008.txt temp.flist | grep 'comp-c\|comp-d' | sort >dev_t.flist
grep -F -f $local/nbest-dev-2008.txt temp.flist | grep -v 'comp-c\|comp-d' | sort >dev_s.flist
rm -f temp.flist

# create utt2spk, spk2utt, txt, segments, scp, spk2gender
for x in train_t train_s dev_t dev_s; do
	$local/process_flist.pl $cgn $x
	recode -d h..u8 $x.txt					# CGN is not in utf-8 by default
	cat $x.utt2spk | $utils/utt2spk_to_spk2utt.pl > $x.spk2utt || exit 1;
done

# prepare lexicon
## If you have a lexicon prepared, you can simply place it in $dictdir and it will be used instead of the default CGN one
if [ ! -f $dictdir/lexicon.txt ]; then
	mkdir -p $dictdir
	cat $cgn/data/lexicon/xml/cgnlex.lex | recode -d h..u8 | perl -CSD $local/format_lexicon.pl $lang | sort >$dictdir/lexicon.txt
	
	# Replace the line above with this one for the CGN corpus version 2.x.x
	# cat $cgn/data/lexicon/xml/cgnlex_2.0.lex | recode -d h..u8 | perl -CSD $local/format_lexicon.pl $lang | sort >$dictdir/lexicon.txt

	## uncomment lines below to convert to UTwente phonetic lexicon	
	# cp $dictdir/lexicon.txt $dictdir/lexicon.orig.txt	
	# cat $dictdir/lexicon.orig.txt | perl $local/cgn2nbest_phon.pl >$dictdir/lexicon.txt
fi
if ! grep -q "^<unk>" $dictdir/lexicon.txt; then
	echo -e "<unk>\t[SPN]" >>$dictdir/lexicon.txt
fi
if ! grep -q "^ggg" $dictdir/lexicon.txt; then
	echo -e "ggg\t[SPN]" >>$dictdir/lexicon.txt
fi
if ! grep -q "^xxx" $dictdir/lexicon.txt; then
	echo -e "xxx\t[SPN]" >>$dictdir/lexicon.txt
fi
# the rest
echo SIL > $dictdir/silence_phones.txt
echo SIL > $dictdir/optional_silence.txt
cat $dictdir/lexicon.txt | awk -F'\t' '{print $2}' | sed 's/ /\n/g' | sort | uniq >$dictdir/nonsilence_phones.txt
touch $dictdir/extra_questions.txt
rm -f $dictdir/lexiconp.txt

cd $base
$utils/prepare_lang.sh $dictdir "<unk>" data/local/lang_tmp_nosp data/lang_nosp || exit 1;

# move everything to the right place
for x in train_t train_s dev_t dev_s; do
	mkdir -p data/$x
	cp $dir/${x}_wav.scp data/$x/wav.scp || exit 1;
	cp $dir/$x.txt data/$x/text || exit 1;
	cp $dir/$x.spk2utt data/$x/spk2utt || exit 1;
	cp $dir/$x.utt2spk data/$x/utt2spk || exit 1;
	cp $dir/$x.segments data/$x/segments || exit 1;
	$utils/filter_scp.pl data/$x/spk2utt $dir/${x}.spk2gender > data/$x/spk2gender || exit 1;
	$utils/fix_data_dir.sh data/$x || exit 1;
done

echo "Data preparation succeeded"
