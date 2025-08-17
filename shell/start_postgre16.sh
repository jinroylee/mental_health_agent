docker run -d --name mh-pg \
  -e POSTGRES_USER=mh_user \
  -e POSTGRES_PASSWORD=mh_pass \
  -e POSTGRES_DB=mh_graph \
  -p 5432:5432 \
  -v /Users/user/Desktop/app/personal/mental_health_agent/storage/pgdata:/var/lib/postgresql/data \
  postgres:16

docker run -d --name mh-pg \
  -e POSTGRES_USER=mh_user \
  -e POSTGRES_PASSWORD=mh_pass \
  -e POSTGRES_DB=mh_graph \
  -p 5432:5432 \
  -v /Users/jinholee/Projects/mental_health_agent/storage/pgdata:/var/lib/postgresql/data \
  postgres:16