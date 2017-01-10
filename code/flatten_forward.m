function [ after ] = flatten_forward(layer, before, after)
%FLATTEN_FORWARD Forward flattening method

data = before.x;
shape = size(data);
data = permute(data,[3,2,1,4]);
output = reshape(data,[1,1,shape(1)*shape(2)*shape(3), shape(4)]);
after.x = output;
end

