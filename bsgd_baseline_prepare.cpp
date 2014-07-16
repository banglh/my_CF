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
mutex mymutex;

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
		training_rmse(iteration, gcontext);
		run_validation(pvalidation_engine, gcontext);
	}

	/**
	 *  Vertex update function.
	 */
	void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex,
			graphchi_context &gcontext) {
		// if it's baseline initialization
		if (!baselineInitFinished) {
			// if this is an user vertex
			mymutex.lock();
			if (vertex.num_outedges() > 0) {
				// calculate this user's average rating
				vid_t userId = vertex.id();
				int nUrated = vertex.num_edges();
				float avgUrated = 0.0;
				// for each item this user rated
				for (int e = 0; e < nUrated; e++) {
					avgUrated += vertex.edge(e)->get_data();
					avgRating += vertex.edge(e)->get_data();
				}
				// store this user's average rating
				userBias[userId] = avgUrated / nUrated;
				nUserTrain += 1;
				updatedVertexNum += 1;
			} else if (vertex.num_inedges() > 0) {
				// calculate the average rating of this item
				vid_t itemId = vertex.id();
				int nIrated = vertex.num_edges();
				float avgIrated = 0.0;
				// for each user has rated this item
				for (int e = 0; e < nIrated; e++) {
					avgIrated += vertex.edge(e)->get_data();
				}
				// store the average rating of this item
				itemBias[itemId - M] = avgIrated / nIrated;
				nItemTrain += 1;
				updatedVertexNum += 1;
			} else {
				updatedVertexNum += 1;
			}
			// check if this is the last vertex
			if (updatedVertexNum == (M + N)) {
				baselineInitFinished = true;
				// calculate the average rating
				avgRating = avgRating / L;

				// for users and items have not (been) rated, bias = average rating
				for (int u = 0; u < M; u++) {
					if (userBias[u] == 0.0)
						userBias[u] = avgRating;
				}
				for (int i = 0; i < N; i++) {
					if (itemBias[i] == 0.0)
						itemBias[i] = avgRating;
				}

				// calculate user biases and items biases
				vec avgRatingUVec = ones(M);
				vec avgRatingIvec = ones(N);
				avgRatingUVec = avgRatingUVec * avgRating;
				avgRatingIvec = avgRatingIvec * avgRating;
				userBias = userBias - avgRatingUVec;
				itemBias = itemBias - avgRatingIvec;

				// fill in the latent features vectors
				// P
				float tmp = avgRating / (D - 2);
				cout << "muy/(k-2) = " << tmp << endl;
				for (int u = 0; u < M; u++) {
					latent_factors_inmem[u].pvec[D - 1] = 1;
					latent_factors_inmem[u].pvec[D - 2] = userBias[u];
					for (int d = 0; d < D - 2; d++) {
						latent_factors_inmem[u].pvec[d] = tmp;
					}
				}

				// Q
				for (int i = M; i < M + N; i++) {
					latent_factors_inmem[i].pvec[D - 1] = itemBias[i - M];
					for (int d = 0; d < D - 1; d++) {
						latent_factors_inmem[i].pvec[d] = 1;
					}
				}
			}
			mymutex.unlock();
		}

		// otherwise update vertex as usual
		else {
		}
	}
};

//dump output to file
void output_sgd_result(std::string filename) {
	MMOutputter_mat<vertex_data> user_mat(filename + "-baseline_P.mm", 0, M,
			"This file contains SGD output matrix P. In each row D factors of a single user node.",
			latent_factors_inmem);
	MMOutputter_mat<vertex_data> item_mat(filename + "-baseline_Q.mm", M, M + N,
			"This file contains SGD  output matrix Q. In each row D factors of a single item node.",
			latent_factors_inmem);

	logstream(LOG_INFO) << "SGD output files (in matrix market format): "
			<< filename << "-baseline_P.mm" << ", "
			<< filename + "-baseline_Q.mm " << std::endl;
}

int main(int argc, const char ** argv) {

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
	initType = "baseline";
	userBias = zeros(M);
	itemBias = zeros(N);
	nUserTrain = 0;
	nItemTrain = 0;
	avgRating = 0.0;
	updatedVertexNum = 0;
	baselineInitFinished = false;
	init_baseline<std::vector<vertex_data> >(latent_factors_inmem);

	if (validation != "") {
		int vshards = convert_matrixmarket<EdgeDataType>(validation, 0, 0, 3,
				VALIDATION, false);
		init_validation_rmse_engine<VertexDataType, EdgeDataType>(
				pvalidation_engine, vshards, &bsgd_predict);
	}

	bsgd_print_config();

	/* Run */
	SGDVerticesInMemProgram program;
	graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards,
			false, m);
	set_engine_flags(engine);
	pengine = &engine;
	engine.run(program, niters);

	std::cout << "nUserTrain: " << nUserTrain << std::endl;
	std::cout << "nItemTrain: " << nItemTrain << std::endl;
	std::cout << "updatedVertexNum: " << updatedVertexNum << std::endl;
	std::cout << "avgRating: " << avgRating << std::endl;
	std::cout << "nRating: " << L << std::endl;

	/* Output latent factor matrices in matrix-market format */
	output_sgd_result(training);

	/* Report execution metrics */
	if (!quiet)
		metrics_report(m);

	return 0;
}
