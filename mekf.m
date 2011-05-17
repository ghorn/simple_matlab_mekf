function mekf() %#eml
    % this is a test of the below functions and half hearted 
    % right now it just checks

    mekf_config = get_mekf_config();
    mekf_state = init_mekf_state();
    
    dt = 0.01;
    
    for k = 1:10
	gyros  = randn(3,1);
	mags   = randn(3,1);
	accels = randn(3,1);
	
	%%%%%%%%%%%%%%%%%%%%%%%% this is how to run the filter  %%%%%%%%%%%%%%%%%%%%%%%%%
	% magnetometer update (if mags available at this time step)
	mekf_state = mekf_vec_update( mekf_state, mekf_config.b_field_ned, mags, mekf_config.R_mags);
	
        % accelerometer update (if accels available at this time step)
        % This implementation de-weights R_accels when norm(accels) is far from one g.
        % This performs pretty well unless you are flying in a constant circle.
        % In that case, you might try estimating inertial acceleration and subracting it out
        % by differentiating GPS or running a translational filter in parallel.
        % The best solution is to have a combined translational/attitude filter which uses gps updates,
        % but that is computationally expensive and this filter is intended to be relatively cheap.
        R_accels = accel_R_heuristic(accels, mekf_config, dt);
        mekf_state = mekf_vec_update( mekf_state, mekf_config.g_ned, accels, R_accels);
        
        % integration (assume gyros are available at the fundamental sample time)
        mekf_state = mekf_integrate(mekf_state, mekf_config, gyros, dt);
        mekf_state.P = mekf_propogate_covariance( mekf_state.P, dt, mekf_config);
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
end

function R_accels = accel_R_heuristic( accels, mekf_config, dt )
    
    % deweigh R_accels when norm(accels) is far from 9.8

    % lowpassing error helps a bit
    persistent accel_error_lp;
    if isempty(accel_error_lp)
	accel_error_lp = 0;
    end

    accel_error = norm(accels) - norm(mekf_config.g_ned);
    accel_error_lp = simple_lowpass(dt, mekf_config.accel_heuristic_tau, accel_error_lp, accel_error);

    dR = mekf_config.accel_heuristic_gain * accel_error_lp * accel_error_lp;
    R_accels = diag((mekf_config.accel_heuristic_R0 + dR)*ones(3,1));

end

function state = simple_lowpass(dt, tau, state, input)
    emdt = exp(-dt/tau);
    state = state*emdt + input*(1-emdt);
end


function mekf_config = get_mekf_config()
    % acceleration due to gravity
    mekf_config.g_ned = [ 0.0;
		          0.0;
			 -9.8;];
    
    % magnetic field in NED
    % should be normalized to 1
    mekf_config.b_field_ned = [0.475358676694968;
			       0.118091726731412;
			       0.871830529729490;];

    % gyro wideband variance
    mekf_config.gyro_wb_var = [0.012;
			       0.012;
			       0.012;].^2;

    % accelerometer R deweighing heuristic
    mekf_config.accel_heuristic_gain = 250;
    mekf_config.accel_heuristic_R0 = 1.0;
    mekf_config.accel_heuristic_tau = 0.05;

    % magnetometer wideband covariance
    mag_wb_var = [0.5;
		  0.5;
		  0.5;].^2;
    mekf_config.R_mags = diag(mag_wb_var);

    % gyro gauss markov (bias drift) variance
    mekf_config.gyro_gauss_markov_var = [0.0008;
					 0.0008;
					 0.0008;].^2;

    % gyro gauss markov (bias drift) correlation time
    mekf_config.gyro_gauss_markov_tau = [16.0;
					 16.0;
					 16.0;];

    % body to imu rotation
    mekf_config.q_b2m = [1;0;0;0];
end

function mekf_state = init_mekf_state()
    % could do vector matching here, but setting decent initial P accomplishes the same goal
    mekf_state.q_n2m = [1.0;
			0.0;
			0.0;
			0.0;];
    
    mekf_state.gyro_bias = [0;
			    0;
			    0;];
    
    mekf_state.P = diag([1.0*ones(1,3), 0.001*ones(1,3)]);

    mekf_state.time = 0;
end


function mekf_state = mekf_vec_update( mekf_state, v_ned_expected, v_imu_observed, R)

    % linearized measurement matrix H
    H = mekf_H_vec( mekf_state.q_n2m, v_ned_expected);

    % Kalman gain
    % don't forget to use transpose() instead of ()' because ()' is complex
    K = mekf_state.P*transpose(H)/(H*mekf_state.P*transpose(H)+R);
%    K = mekf_state.P*transpose(H)*cholesky_inverse(H*mekf_state.P*transpose(H)+R)
    
    % measurement error
    v_imu_expected = quat2dcm(mekf_state.q_n2m')*v_ned_expected;
    delta_z = v_imu_observed - v_imu_expected;

    % state error/error quaternion
    delta_x = K*delta_z;

    % state update
    q_error = [1.0; delta_x(1:3)];
    mekf_state.q_n2m = quatmultiply(mekf_state.q_n2m', q_error')';
    mekf_state.gyro_bias = mekf_state.gyro_bias + delta_x(4:6);

    % covariance update
    mekf_state.P = K*R*transpose(K) + (eye(6) - K*H)*mekf_state.P*transpose(eye(6)-K*H);

    % (optional, balance covariance matrix)
    mekf_state.P = 0.5*(mekf_state.P + transpose(mekf_state.P));
end

%function Minv = cholesky_inverse(M)
%    L = chol(H, 'lower');
%%    Lt = ctranspose(L);
%    assert(isreal, L);
%    Lt = transpose(L); % can I do this?
%    Minv = inv(L*Lt);
%end

function H = mekf_H_vec(q_n2m, v_ned_expected)

  % dh/d(xhat_predicted)
  % the "vec" in "H_vec" means that we can use this same function for gravity [0,0,-9.8] or B field [Bx, By, Bz]

  q0 = q_n2m(1);
  q1 = q_n2m(2);
  q2 = q_n2m(3);
  q3 = q_n2m(4);

  vx = v_ned_expected(1);
  vy = v_ned_expected(2);
  vz = v_ned_expected(3);

  b_01 = - 4*(q0*q2 + q1*q3)*vx ...
         + 4*(q0*q1 - q2*q3)*vy ...
         - 2*(q0*q0 - q1*q1 - q2*q2 + q3*q3)*vz;

  b_02 = - 4*(q0*q3 - q1*q2)*vx ...
         + 4*(q0*q1 + q2*q3)*vz ...
         + 2*(q0*q0 - q1*q1 + q2*q2 - q3*q3)*vy;

  b_12 = - 4*(q0*q3 + q1*q2)*vy ...
         + 4*(q0*q2 - q1*q3)*vz ...
         - 2*(q0*q0 + q1*q1 - q2*q2 - q3*q3)*vx;

  H = [    0,  b_01, b_02,  0, 0, 0;
       -b_01,     0, b_12,  0, 0, 0;
       -b_02, -b_12,    0,  0, 0, 0;];

end

function P = mekf_propogate_covariance(P, dt, mekf_config)

    % try using sparse matrices here - see if it's efficient and/or eml compliant
    F = diag([-0.5*dt;
	      -0.5*dt;
	      -0.5*dt;
	      exp(-dt/mekf_config.gyro_gauss_markov_tau(1));
	      exp(-dt/mekf_config.gyro_gauss_markov_tau(2));
	      exp(-dt/mekf_config.gyro_gauss_markov_tau(3));]);

    % P = F*P*F^T
    P = F*P*transpose(F);

    % P += G*Q*G^T
    P(1,1) = P(1,1) + 0.25*dt*dt*mekf_config.gyro_wb_var(1);
    P(2,2) = P(2,2) + 0.25*dt*dt*mekf_config.gyro_wb_var(2);
    P(3,3) = P(3,3) + 0.25*dt*dt*mekf_config.gyro_wb_var(3);
    P(4,4) = P(4,4) + 2*dt/mekf_config.gyro_gauss_markov_tau(1)*mekf_config.gyro_gauss_markov_var(1);
    P(5,5) = P(5,5) + 2*dt/mekf_config.gyro_gauss_markov_tau(2)*mekf_config.gyro_gauss_markov_var(2);
    P(6,6) = P(6,6) + 2*dt/mekf_config.gyro_gauss_markov_tau(3)*mekf_config.gyro_gauss_markov_var(3);

end

function mekf_state = mekf_integrate( mekf_state, mekf_config, gyros_in_imu, dt)

    % time
    mekf_state.time = mekf_state.time + dt;

    % rotate quaternion by angular velocity
    w_mi_i = gyros_in_imu - mekf_state.gyro_bias;

    w_norm = norm( w_mi_i ) + 1.0e-12;
    q_diff = [cos(0.5*w_norm*dt);
	      sin(0.5*w_norm*dt).*w_mi_i./w_norm;];

    mekf_state.q_n2m = quatmultiply( mekf_state.q_n2m', q_diff')';
    
    % let gyro biases decay
    % only need to compute these exponentials once if they're not changing
    mekf_state.gyro_bias = mekf_state.gyro_bias .*[exp(-dt/mekf_config.gyro_gauss_markov_tau(1));
						   exp(-dt/mekf_config.gyro_gauss_markov_tau(2));
						   exp(-dt/mekf_config.gyro_gauss_markov_tau(3));];

end
