from evaluate import predict_and_evaluate, evaluate_by_file

if(len(sys.argv) > 2):
	evaluate_by_file(sys.argv[1], sys.argv[2])
else:
	predict_and_evaluate(sys.argv[1])