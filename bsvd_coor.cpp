/**
 * @file
 * @author  Danny Bickson, based on code by Aapo Kyrola
 * @version 1.0
 *
 * @section LICENSE
 *
 * Copyright [2012] [Carnegie Mellon University]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 *
 * @section DESCRIPTION
 *
 * Matrix factorization with the Alternative Least Squares (ALS) algorithm.
 * See the papers:
 * H.-F. Yu, C.-J. Hsieh, S. Si, I. S. Dhillon, Scalable Coordinate Descent Approaches to Parallel Matrix Factorization for Recommender Systems. IEEE International Conference on Data Mining(ICDM), December 2012.
 * Steffen Rendle, Zeno Gantner, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2011. Fast context-aware recommendations with factorization machines. In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval (SIGIR '11). ACM, New York, NY, USA, 635-644. *
 */

#include "eigen_wrapper.hpp"
#include "common.hpp"

struct vertex_data {
	vec pvec;

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

/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program.
 */
typedef vertex_data VertexDataType;
typedef float EdgeDataType;  // Edges store the "rating" of user->movie pair

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL;
graphchi_engine<VertexDataType, EdgeDataType> * pvalidation_engine = NULL;
std::vector<vertex_data> latent_factors_inmem;

#include "io.hpp"
#include "rmse.hpp"
#include "rmse_engine.hpp"

/** compute a missing value based on ALS algorithm */
float bsvd_predict(const vertex_data& user, const vertex_data& movie,
		const float rating, double & prediction, void * extra = NULL) {

	prediction = dot_prod(user.pvec, movie.pvec);
	//truncate prediction to allowed values
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
struct ALSVerticesInMemProgram: public GraphChiProgram<VertexDataType,
		EdgeDataType> {
	/**
	 *  Vertex update function
	 */
	void update(graphchi_vertex<VertexDataType, EdgeDataType> &vertex, graphchi_context &gcontext) {
		if (vertex.num_edges() > 0) {
			vertex_data & vdata = latent_factors_inmem[vertex.id()];
			vec R_cache = zeros(vertex.num_edges());

			// for each latent feature
			for (int x = 0; x < D; x++) {
				double numerator = 0.0;
				double denominator = 0.0;
				bool compute_rmse = (vertex.num_outedges() > 0 && x == 0);

				// for each user/item have connection with this node
				for (int j = 0; j < vertex.num_edges(); j++) {
					float observation = vertex.edge(j)->get_data();	// rating value of this connection
					vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(j)->vertex_id()];// latent feature vector of this node
					double prediction;
					double rmse = 0;

					// if x = 0, calculate rmse and initialize R_cache
					if (x == 0) {
						rmse = bsvd_predict(vdata, nbr_latent, observation, prediction);
						R_cache[j] = observation - prediction;
					}

					//** B term
					double B = exp(alpha * (prediction - maxval));
					//** C term
					double C = exp(alpha * (minval - prediction));

					// compute numerator
					numerator += 2 * nbr_latent.pvec[x] * (R_cache[j] + vdata.pvec[x] * nbr_latent.pvec[x])
								- lambda * alpha * nbr_latent.pvec[x] * (B - C)
								+ lambda * alpha * alpha * nbr_latent.pvec[x] * nbr_latent.pvec[x] * vdata.pvec[x] * (B + C);

					//compute denominator
					denominator += nbr_latent.pvec[x] * nbr_latent.pvec[x] * (2 + lambda * alpha * alpha * (B + C));

					//record rmse
					if (compute_rmse)
						rmse_vec[omp_get_thread_num()] += rmse;
				}

				// update if denominator != 0
				if (denominator != 0.0) {
					double z = numerator / denominator;
					vec old = vdata.pvec;

					for (int j = 0; j < vertex.num_edges(); j++) {
						vertex_data & nbr_latent = latent_factors_inmem[vertex.edge(j)->vertex_id()];
						//update using equation (7) in ICDM paper
						//R_ij -= (z - w_it )*h_jt
						R_cache[j] -= ((z - old[x]) * nbr_latent.pvec[x]);
					}

					//update using equation (8) in ICDM paper
					//w_it = z;
					vdata.pvec[x] = z;
				}
			}
		}
	}

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
};

void output_als_result(std::string filename) {
	MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M,
			"This file contains ALS output matrix U. In each row D factors of a single user node.",
			latent_factors_inmem);
	MMOutputter_mat<vertex_data> item_mat(filename + "_V.mm", M, M + N,
			"This file contains ALS  output matrix V. In each row D factors of a single item node.",
			latent_factors_inmem);
	logstream(LOG_INFO) << "ALS output files (in matrix market format): "
			<< filename << "_U.mm" << ", " << filename + "_V.mm " << std::endl;
}

int main(int argc, const char ** argv) {

	print_copyright();

	/* GraphChi initialization will read the command line
	 arguments and the configuration file. */
	graphchi_init(argc, argv);

	/* Metrics object for keeping track of performance counters
	 and other information. Currently required. */
	metrics m("bsvd_coor-inmemory-factors");

	/* Basic arguments for application. NOTE: File will be automatically 'sharded'. */
	alpha = get_option_float("alpha", 1.0);
	lambda = get_option_float("lambda", 1.0);

	parse_command_line_args();
	parse_implicit_command_line();

	/* Preprocess data if needed, or discover preprocess files */
	int nshards = convert_matrixmarket<EdgeDataType>(training, 0, 0, 3, TRAINING, false);

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
		init_validation_rmse_engine<VertexDataType, EdgeDataType>(pvalidation_engine, vshards, &bsvd_predict);
	}

	/* load initial state from disk (optional) */
	if (load_factors_from_file) {
		load_matrix_market_matrix(training + "_U.mm", 0, D);
		load_matrix_market_matrix(training + "_V.mm", M, D);
	}

	/* Run */
	ALSVerticesInMemProgram program;
	graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m);
	set_engine_flags(engine);
	pengine = &engine;
	engine.run(program, niters);

	/* Output latent factor matrices in matrix-market format */
	output_als_result(training);
	test_predictions(&bsvd_predict);

	/* Report execution metrics */
	if (!quiet)
		metrics_report(m);

	return 0;
}
