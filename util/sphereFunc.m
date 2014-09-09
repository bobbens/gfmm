function [ logfunc, expfunc ] = sphereFunc()

   logfunc = @s_logfunc;
   expfunc = @s_expfunc;

   function u = s_logfunc( data, mu )
      if norm(mu-[0;0;-1])<1e-6;
         R = [1 0 0; 0 -1 0; 0 0 -1];
      else
         R = rotM(mu);
      end
      u = LogNP(R*data')';
   end

   function x = s_expfunc( data, mu )
      x = (inv(rotM(mu))*ExpNP(data'))';
   end
end

