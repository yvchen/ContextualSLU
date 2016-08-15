DOMAIN=cortana.communication.5
DATADIR=data
PROG=program/SequenceTagger.py
TRAIN=$DATADIR/"$DOMAIN".train.iob
TEST=$DATADIR/"$DOMAIN".test.iob
DEV=$DATADIR/"$DOMAIN".dev.iob
RES_PATH=sample
MDL_PATH=sample
OPTFUNC=adam
if [ ! $1 ]; then
	echo "Usage: $0 <model (e.g. rnn | igru | memn2n-c-gru)> <backend (theano | tensorflow)> <gpu id>"
else
	if [ ! $2 ]; then
		echo "Default is CPU for theano or automatic setting for tensorflow depending on what the backend is using."
		GPUSET=''
	else
		if [ ! $3 ]; then
			echo "Usage: $0 <model (e.g. rnn | igru | memn2n-c-gru)> <backend (theano | tensorflow)> <gpu id>"
			exit
		else
			GPUSET="THEANO_FLAGS=device=gpu"$3",floatX=float32"
		fi
	fi
	MDL=$1
	CMD="$GPUSET python $PROG --train $TRAIN --dev $DEV --test $TEST --sgdtype $OPTFUNC --arch $MDL --iter_per_epoch 30 --hidden_size 100 --embedding_size 150 --out $RES_PATH -m 3 --mdl_path $MDL_PATH --record_epoch 1 --his_length 4 --time_length 50 --dropout True --dropout_ratio 0.5 --input_type embedding"
	echo $CMD
fi
