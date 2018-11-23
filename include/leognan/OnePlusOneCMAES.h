#ifndef LEOGNAN_ONE_PLUS_ONE_CMAES_H
#define LEOGNAN_ONE_PLUS_ONE_CMAES_H

#include <leognan/Randomizer.h>
#include <leognan/FitnessFunction.h>



namespace leognan {
	class OnePlusOneCMAES {
	public:
		void
		init(const FitnessFunction* fitness_func,
		     const Eigen::VectorXd& X_init) {
			int n = fitness_func->get_dimension();

			// Allocate ressources
			m_A = Eigen::MatrixXd::Zero(n, n);
			m_C = Eigen::MatrixXd::Identity(n, n);
			m_Z = Eigen::VectorXd::Zero(n);
			m_Y = Eigen::VectorXd::Zero(n);
			m_pc = Eigen::VectorXd::Zero(n);
			m_X_parent = X_init;
			m_X_offspring = Eigen::VectorXd(n);		

			// Step size control parameters
			m_d = 1. + .5 * n;
			m_psucc_target = 2. / 11.;
			m_psucc = m_psucc_target;
			m_cp = 1. / 12.;

			// Covariance control parameters
			m_cc = 2. / (n + 2.);
			m_ccov = 2. / (n * n + 6.);
			m_pthreshold = .44;

			// Evaluate initial solution
			m_sigma = 1.;
			m_f_parent = fitness_func->get_fitness(X_init);
		}

		void
		update(Randomizer& rnd,
		       const FitnessFunction* fitness_func) {
			// Compute A
			m_A = m_C.llt().matrixL();

			// Compute a step
			for(Eigen::Index i = 0; i < m_Z.rows(); ++i)
				m_Z(i) = rnd.next_gaussian(1.);

			// Compute offspring
			m_Y = m_A * m_Z;
			m_X_offspring = m_X_parent + m_sigma * m_Y;

			// Evaluate offspring
			double f_offspring = fitness_func->get_fitness(m_X_offspring);
			bool success = f_offspring < m_f_parent;

			// Update step size
			update_step_size(success);

			// Update covariance matrix and parent
			if (success) {
				m_f_parent = f_offspring;
				m_X_parent = m_X_offspring;
				update_covariance(m_Y);
			}
		}
 
	private:
		void
		update_step_size(bool success) {
			m_psucc = (1. - m_cp) * m_psucc + m_cp * success;
			m_sigma *= std::exp((m_psucc - (m_psucc_target / (1. - m_psucc_target)) * (1. - m_psucc)) / m_d);
		}

		void
		update_covariance(const Eigen::VectorXd& Y) {
			if (m_psucc < m_pthreshold) {
				m_pc = (1. - m_cc) * m_pc + std::sqrt(m_cc * (2. - m_cc)) * Y;
				m_C = (1. - m_ccov) * m_C + m_ccov * m_pc * m_pc.transpose();
			}
			else {
				m_pc = (1. - m_cc) * m_pc;
				m_C = (1. - m_ccov) * m_C + (m_ccov * m_pc * m_pc.transpose() + m_cc * (2. - m_cc) * m_C);
			}
		}

		double m_cp;
		double m_cc;
		double m_ccov;
		double m_pthreshold;
		double m_sigma;
		double m_psucc;
		double m_psucc_target;
		double m_lambda_succ;
		double m_d;
		double m_f_parent;

		Eigen::MatrixXd m_A;
		Eigen::MatrixXd m_C;
		Eigen::VectorXd m_Z;
		Eigen::VectorXd m_Y;
		Eigen::VectorXd m_pc;
		Eigen::VectorXd m_X_parent;
		Eigen::VectorXd m_X_offspring;

	}; // class OnePlusOneCMAES
} // namespace leognan



#endif // LEOGNAN_ONE_PLUS_ONE_CMAES_H
