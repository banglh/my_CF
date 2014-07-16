/**
 * @file
 * @author  mr_ice
 * @version 1.0
 *
 *
 */

#include <iostream>
#include <fstream>
#include "eigen_wrapper.hpp"
#include "common.hpp"
#include <ctime>
#include <memory>
#include <set>

using namespace std;

std::ofstream logFile; // output file

struct vertex_data {
	vec pvec; //storing the feature vector

	vertex_data() {
		pvec = zeros(D);
	}
	void set_val(int index, float val) {
		pvec[index] = val;
	}
	float get_val(int index) {
		return pvec[index];
	}

};

#include "util.hpp"

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program.
 */
typedef vertex_data VertexDataType;
typedef float EdgeDataType; // Edges store the "rating" of user->item pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL;
graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine = NULL;
std::vector<vertex_data> latent_factors_inmem;

#include "rmse.hpp"
#include "rmse_engine.hpp"
#include "io.hpp"

/** compute a missing value based on SGD algorithm */
float bsgd_predict(const vertex_data& user, const vertex_data& item,
		const float rating, double & prediction, void * extra = NULL) {

	prediction = dot_prod(user.pvec, item.pvec);

//  //truncate prediction to allowed values
//  prediction = std::min((double)prediction, maxval);
//  prediction = std::max((double)prediction, minval);
	//return the squared error

	float err = rating - prediction;
	assert(!std::isnan(err));
	return err * err;

}

/**
 * GraphChi programs need to subclass GraphChiProgram<vertex-type, edge-type>
 * class. The main logic is usually in the update function.
 */
struct SGDVerticesInMemProgram: public GraphChiProgram<VertexDataType,
		EdgeDataType> {

	/**
	 * Called before an iteration is started.
	 */
	void before_iteration(int iteration, graphchi_context &gcontext) {
		reset_rmse(gcontext.execthreads);
	}

	/**
	 * Called after an iteration has finished.
	 */
	void after_iteration(int iteration, graphchi_context &gcontext) {
		muy *= step_dec;
		training_rmse(iteration, gcontext);
		run_validation(pvalidation_engine, gcontext);
	}

	/**
	 *  Vertex update function.
	 */
	void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex,
			graphchi_context &gcontext) {

		// if this is user vertex
		if (vertex.num_outedges() > 0) {
			vertex_data & user = latent_factors_inmem[vertex.id()];
			std::set<int> ratedItems; // set of rated items
			// for each item this user rated
			for (int e = 0; e < vertex.num_edges(); e++) {
				// get rating value
				float observation = vertex.edge(e)->get_data();

				// get rated item
				int itemId = vertex.edge(e)->vertex_id();
				ratedItems.insert((int) itemId);
				vertex_data & item = latent_factors_inmem[itemId];

				// calculate the current estimation for the rating value
				double estScore;
				rmse_vec[omp_get_thread_num()] += bsgd_predict(user, item,
						observation, estScore);

				// calculate the error
				double err = observation - estScore;
				if (std::isnan(err) || std::isinf(err))
					logstream(LOG_FATAL)
							<< "SGD got into numerical error. Please tune step size using --muy and --alpha"
							<< std::endl;

				//NOTE: the following code is not thread safe, since potentially several
				//user nodes may updates this item gradient vector concurrently. However in practice it
				//did not matter in terms of accuracy on a multicore machine.
				//if you like to defend the code, you can define a global variable
				//mutex mymutex;
				//
				// update the feature vectors
				double expTerm = exp(alpha * (estScore - maxval))
						- exp(alpha * (minval - estScore));
				double temp = muy * (err - lambda * alpha * expTerm);
				//and then do: mymutex.lock()
				user.pvec += temp * item.pvec;
				//and here add: mymutex.unlock();
				item.pvec += temp * user.pvec;
			}
			// if the bound constraint is for the whole rating matrix
			if (bsgd_ver == 2) {
				// for each item has not been rated by the user
				for (int i = M; i < M + N; i++) {
					if (ratedItems.count(i) == 0) {
						// calculate the current estimation for the rating value
						vertex_data & item = latent_factors_inmem[i];
						double estScore = dot_prod(user.pvec, item.pvec);
						// update feature vectors
						double expTerm = exp(alpha * (estScore - maxval))
								- exp(alpha * (minval - estScore));
						user.pvec -= muy * lambda * alpha * expTerm * item.pvec;
						item.pvec -= muy * lambda * alpha * expTerm * user.pvec;
					}
				}
			}
		}
	}
};

//dump output to file
void output_sgd_result(std::string filename) {
	MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M,
			"This file contains SGD output matrix U. In each row D factors of a single user node.",
			latent_factors_inmem);
	MMOutputter_mat<vertex_data> item_mat(filename + "_V.mm", M, M + N,
			"This file contains SGD  output matrix V. In each row D factors of a single item node.",
			latent_factors_inmem);

	logstream(LOG_INFO) << "SGD output files (in matrix market format): "
			<< filename << "_U.mm" << ", " << filename + "_V.mm " << std::endl;
}

void print_config_to_file(std::ofstream & f) {
	f << "****** BSGD experiment *******" << std::endl;
	f << "+ Dataset: " << dataset << std::endl;
	f << "+ # Users: " << M << std::endl;
	f << "+ # Items: " << N << std::endl;
	f << "+ Min rating: " << minval << std::endl;
	f << "+ Max rating: " << maxval << std::endl;
	f << "+ Training ratings: " << L << std::endl;
	f << "*** Parameters setting ***" << std::endl;
	f << "	+ BSGD version: " << bsgd_ver << std::endl;
	f << "	+ # latent features (K): " << D << std::endl;
	switch (init_features_type) {
	case 1: // bounded random
		f << "	+ Init features type: [bounded random]" << std::endl;
		break;
	case 2: // baseline
		f << "	+ Init features type: [baseline]" << std::endl;
		break;
	case 3: // random
		f << "	+ Init features type: [random]" << std::endl;
		break;
	default: // random
		f << "	+ Init features type: [random]" << std::endl;
	}
	f << "	+ Experiment: " << experiment << std::endl;
	f << "	+ Run: " << run << std::endl;
	f << "	+ lambda = " << lambda << std::endl;
	f << "	+ alpha = " << alpha << std::endl;
	f << "	+ muy = " << muy << std::endl;
	f << "	+ step_dec = " << step_dec << std::endl;
	f << "	+ max iterations = " << niters << std::endl;
	f << "	+ halt_on_rmse_increase = " << halt_on_rmse_increase << std::endl;
	f << "	+ halt_on_minor_improvement = " << halt_on_minor_improvement
			<< std::endl;
}

int main(int argc, const char ** argv) {

	bool logMode = true;

	//* GraphChi initialization will read the command line arguments and the configuration file. */
	graphchi_init(argc, argv);

	/* Metrics object for keeping track of performance counters
	 and other information. Currently required. */
	metrics m("bsgd-inmemory-factors");

	/* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
	muy = get_option_float("muy", 5e-3);
	alpha = get_option_float("alpha", 1.0);
	lambda = get_option_float("lambda", 1.0);
	step_dec = get_option_float("step_dec", 1);

	parse_command_line_args();
	parse_implicit_command_line();

	/* Preprocess data if needed, or discover preprocess files */
	int nshards = convert_matrixmarket<EdgeDataType>(training, 0, 0, 3,
			TRAINING, false);

	// initialize features vectors
	std::string initType;
	switch (init_features_type) {
	case 1: // bounded random
		// randomly initialize feature vectors so that rmin < rate < rmax
		initType = "bounded-random";
		init_random_bounded<std::vector<vertex_data> >(latent_factors_inmem, !load_factors_from_file);
		break;
	case 2: // baseline
		initType = "baseline";
		init_baseline<std::vector<vertex_data> >(latent_factors_inmem);
		load_matrix_market_matrix(training + "-baseline_P.mm", 0, D);
		load_matrix_market_matrix(training + "-baseline_Q.mm", M, D);
		break;
	case 3: // random
		initType = "random";
		init_feature_vectors<std::vector<vertex_data> >(M + N, latent_factors_inmem, !load_factors_from_file);
		break;
	default: // random
		initType = "random";
		init_feature_vectors<std::vector<vertex_data> >(M + N, latent_factors_inmem, !load_factors_from_file);
	}

	if (validation != "") {
		int vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3, VALIDATION, false);
		init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &bsgd_predict);
	}

//	/* load initial state from disk (optional) */
//	if (load_factors_from_file) {
//		load_matrix_market_matrix(training + "_U.mm", 0, D);
//		load_matrix_market_matrix(training + "_V.mm", M, D);
//	}

	bsgd_print_config();

	std::streambuf *fileBuf, *backup;
	std::string fileName;
	if (logMode) {
		// create log file
		std::stringstream fn;
		time_t now = time(0);
		std::string basePath = "./results/";
		fn << basePath << now << "_" << dataset << "_BSGD-" << bsgd_ver << "_"
				<< initType << "_RMSE_K" << D << "_Exp" << experiment << "_Run"
				<< run << ".txt";
		fileName = fn.str();

		logFile.open(fileName.c_str());

		// print config to file
		print_config_to_file(logFile);

		// change the stream buffer
		backup = std::cout.rdbuf();
		fileBuf = logFile.rdbuf();
		std::cout.rdbuf(fileBuf);
	}

	/* Run */
	SGDVerticesInMemProgram program;
	graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m);
	set_engine_flags(engine);
	pengine = &engine;
	engine.run(program, niters);

	/* Output latent factor matrices in matrix-market format */
	output_sgd_result(training);
	double testRMSE = test_predictions(&bsgd_predict);

	// write RMSE on test set to log file
	if (logMode) {
		logFile << "Test RMSE: " << testRMSE << std::endl;
		std::cout.rdbuf(backup);
		logFile.close();
		std::cout << "Finished writing results to file " << fileName
				<< std::endl;
	}

	/* Report execution metrics */
	if (!quiet)
		metrics_report(m);

	return 0;
}
