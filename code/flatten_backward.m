function [ after ] = flatten_backward(layer, after, before)
%FLATTEN_BACKWARD Backward flattening method

data = before.dzdx;
shape = size(data);
output = reshape(data,[64, 7, 7, shape(4)]);
output = permute(output, [3,2,1,4]);
after.dzdx = output;

end
