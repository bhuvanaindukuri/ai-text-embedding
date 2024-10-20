create table documents (
  id serial primary key,
  title text not null,
  body text not null,
  embedding vector(768)
);